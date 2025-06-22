# %% [markdown]
# # Post-Training Quantization (PTQ)
#
# Jupyter-style, but easily runnable as a plain script. **Minimal edits** implement the missing PTQ steps flagged in review – *recursive activation fake‑quant*, *in‑place weight quant*, optional **AdaRound**, and proper ImageNet **validation** evaluation – while touching as little of the original code as possible.

# %% [markdown]
# ## 1. Imports & logging

# %%
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import transforms as T

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. Config dataclass (unchanged, except a switch enabling AdaRound)
# -----------------------------------------------------------------------------


# %%
@dataclass
class Config:
    """Global parameters."""

    seed: int = 42

    class Datasets:
        train_size: int = 2**10
        train_path: str = "./data/training/"

        test_size: int = 2**13
        test_path: str = "./data/testing/"

    class Quant:
        path: str = "./data/quantized_model.pth"

        weight_bits: int = 8  # bit‑width for *all* weights
        act_bits: int = 8  # bit‑width for *all* activations
        per_channel: bool = False  # per‑channel weight quant?

        # Calibration / optimisation options
        mse_samples: int = 16  # batches for activation MSE search
        adaround_iters: int = 1000  # set 0 to disable AdaRound
        adaround_reg: float = 0.01  # rounding regulariser strength


# -----------------------------------------------------------------------------
# 3. Quantiser primitives (unchanged)
# -----------------------------------------------------------------------------

# %%


def symmetric_uniform_quant(t: Tensor, scale: Tensor, bits: int) -> Tensor:
    """Fake‑quantise *t* with a symmetric, uniform mid‑tread quantiser."""
    qmax = 2 ** (bits - 1) - 1  # 127 for 8‑bit
    t_q = torch.round(t / scale).clamp(-qmax - 1, qmax)
    return t_q * scale


def choose_qparams_minmax(t: Tensor, bits: int, reduce_dim: int | None = None) -> Tensor:
    max_val = t.abs().amax(dim=reduce_dim, keepdim=True) if reduce_dim is not None else t.abs().max()
    qmax = 2 ** (bits - 1) - 1
    return (max_val / qmax).clamp(min=1e-8)


def choose_qparams_mse(t: Tensor, bits: int, n_grid: int = 100, reduce_dim: int | None = None) -> Tensor:
    if reduce_dim is None:
        t_absmax = t.abs().max().item()
        search_space = torch.linspace(0.5, 1.0, n_grid, device=t.device)
        best_scale, best_err = None, torch.inf
        for r in search_space:
            scale = r * t_absmax / (2 ** (bits - 1) - 1)
            err = F.mse_loss(symmetric_uniform_quant(t, scale, bits), t)
            if err < best_err:
                best_err, best_scale = err, scale
        return torch.as_tensor(best_scale, device=t.device)
    # Per‑channel search
    scales: list[Tensor] = []
    for c in range(t.shape[reduce_dim]):
        idx = [slice(None)] * t.dim()
        idx[reduce_dim] = c
        scales.append(choose_qparams_mse(t[idx], bits, n_grid))
    return torch.stack(scales, dim=reduce_dim)


# -----------------------------------------------------------------------------
# 4. Cross‑Layer Equalisation (unchanged)
# -----------------------------------------------------------------------------

# %%


def cross_layer_equalization(model: nn.Module) -> nn.Module:
    logger.info("Starting cross‑layer equalization …")
    with torch.no_grad():
        mods = list(model.modules())
        for i in range(len(mods) - 1):
            m1, m2 = mods[i], mods[i + 1]
            if not isinstance(m1, (nn.Conv2d, nn.Linear)) or not isinstance(m2, (nn.Conv2d, nn.Linear)):
                continue
            r1 = m1.weight.data.abs().amax(dim=list(range(1, m1.weight.dim())), keepdim=True).clamp(min=1e-8)
            if isinstance(m2, nn.Conv2d):
                r2 = m2.weight.data.abs().amax(dim=[0, 2, 3], keepdim=True)
            else:
                r2 = m2.weight.data.abs().amax(dim=1, keepdim=True)
            r2 = r2.clamp(min=1e-8)
            s = (r1 * r2).sqrt()
            m1.weight.data /= s
            m2.weight.data *= s
            if m1.bias is not None:
                m1.bias.data /= s.squeeze()
    logger.info("Cross‑layer equalization complete.")
    return model


# -----------------------------------------------------------------------------
# 5. AdaRound (unchanged implementation) + helper to run it on Linear layers
# -----------------------------------------------------------------------------


# %%
class AdaRound(nn.Module):
    """Weight‑rounding refinement."""

    def __init__(self, w: Tensor, bits: int = 8, iters: int = 1000, reg: float = 0.01):
        super().__init__()
        self.orig_w = w.detach()
        self.bits, self.iters, self.reg = bits, iters, reg
        self.alpha = nn.Parameter(torch.zeros_like(w))
        self.qmax = 2 ** (bits - 1) - 1
        self.scale = choose_qparams_minmax(self.orig_w, bits)

    # Helpers ------------------------------------------------------------
    def _h(self):
        return torch.sigmoid(self.alpha) * 1.2 - 0.1  # rectified‑sigmoid

    def quant_w(self):
        w_div = self.orig_w / self.scale
        return (w_div.floor() + self._h()).clamp(-self.qmax - 1, self.qmax) * self.scale

    # Main optimisation --------------------------------------------------
    def optimise(self, act: Tensor):
        opt = torch.optim.Adam([self.alpha], lr=1e-2)
        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(self.quant_w() @ act.T, self.orig_w @ act.T)
            reg_term = self.reg * torch.sum(1 - (2 * self._h() - 1).abs())
            (loss + reg_term).backward()
            opt.step()
        return self.quant_w().detach()


def _adaround_linear_layers(model: nn.Module, act: Tensor, cfg: Config.Quant):
    """Run AdaRound on **Linear** layers only (enough for ResNet classifier head)."""
    if cfg.adaround_iters <= 0:
        return
    logger.info("Running AdaRound on linear layers …")
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            ada = AdaRound(
                mod.weight.data,
                bits=cfg.weight_bits,
                iters=cfg.adaround_iters,
                reg=cfg.adaround_reg,
            )
            mod.weight.data.copy_(ada.optimise(act))


# -----------------------------------------------------------------------------
# 6. Activation‑range calibration (unchanged)
# -----------------------------------------------------------------------------

# %%


def calibrate_model(model: nn.Module, loader, cfg: Config.Quant) -> dict[str, Tensor]:
    logger.info("Calibrating activations on %s batches …", cfg.mse_samples)
    act_scales: dict[str, Tensor] = {}

    def _collect(mod: nn.Module, _in, out):
        name = str(mod._quant_name)
        act_scales[name] = choose_qparams_mse(out.detach(), cfg.act_bits)

    hooks = [mod.register_forward_hook(_collect) for mod in model.modules() if isinstance(mod, QuantStub)]

    model.eval()
    with torch.no_grad():
        for i, (x, *_) in enumerate(loader):
            model(x.to(next(model.parameters()).device, non_blocking=True))
            if i + 1 >= cfg.mse_samples:
                break
    for h in hooks:
        h.remove()
    logger.info("Activation calibration finished.")
    return act_scales


# -----------------------------------------------------------------------------
# 7. QuantStub / DeQuantStub (unchanged)
# -----------------------------------------------------------------------------


# %%
class QuantStub(nn.Module):
    def __init__(self, name: str, bits: int = 8):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.bits = bits
        self._quant_name = name

    def forward(self, x: Tensor):  # type: ignore[override]
        return symmetric_uniform_quant(x, self.scale, self.bits)


class DeQuantStub(nn.Module):
    def forward(self, x: Tensor):  # type: ignore[override]
        return x


# -----------------------------------------------------------------------------
# 8. Helper: recursive insertion of Q‑DQ pairs (replaces the old top‑level loop)
# -----------------------------------------------------------------------------

# %%


def _insert_qdq_recursive(parent: nn.Module, cfg: Config.Quant, prefix: str = ""):
    """Wrap every child module with (QuantStub → child → DeQuantStub)."""
    for name, child in list(parent.named_children()):
        # Skip if already wrapped or a quant/dequant itself
        if isinstance(child, (QuantStub, DeQuantStub)):
            continue
        full_name = f"{prefix}{name}"
        q = QuantStub(full_name + ":input", cfg.act_bits)
        dq = DeQuantStub()
        # Recurse **before** wrapping so inner modules are handled only once
        _insert_qdq_recursive(child, cfg, prefix=full_name + ".")
        setattr(parent, name, nn.Sequential(q, child, dq))


# -----------------------------------------------------------------------------
# 9. Helper: in‑place weight quantisation (simulated int8 weights)
# -----------------------------------------------------------------------------

# %%


def _quantise_weights_inplace(model: nn.Module):
    """Replace float32 weights with their fake‑quantised int representation."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "w_scale"):
            m.weight.data.copy_(symmetric_uniform_quant(m.weight.data, m.w_scale, m.quant_bits))


# -----------------------------------------------------------------------------
# 10. PTQ pipeline (original structure kept, with three tiny additions)
# -----------------------------------------------------------------------------

# %%


def ptq_pipeline(model: nn.Module, calib_loader, /, cfg: Config.Quant = Config.Quant()) -> nn.Module:  # noqa: B008
    logger.info("Starting PTQ pipeline …")
    model = deepcopy(model)

    # (1) Cross‑layer equalisation --------------------------------------
    model = cross_layer_equalization(model)

    # (2) Prepare weight metadata & rough scales ------------------------
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.__dict__.update(quant_bits=cfg.weight_bits, per_channel=cfg.per_channel)
            red_dim = 0 if cfg.per_channel else None
            m.register_buffer(
                "w_scale",
                choose_qparams_mse(m.weight.data, cfg.weight_bits, reduce_dim=red_dim),
            )

    # (3) Insert Q‑DQ pairs **recursively** -----------------------------
    _insert_qdq_recursive(model, cfg)

    # (4) Activation calibration ----------------------------------------
    act_scales = calibrate_model(model, calib_loader, cfg)
    for mod in model.modules():
        if isinstance(mod, QuantStub):
            mod.scale.copy_(act_scales[mod._quant_name])

    # (5) Optional AdaRound (linear only) -------------------------------
    first_batch, *_ = next(iter(calib_loader))
    _adaround_linear_layers(model, first_batch.view(first_batch.size(0), -1), cfg)

    # (6) In‑place weight quantisation ----------------------------------
    _quantise_weights_inplace(model)

    # Store activation scales for export/debugging
    model.__dict__["activation_scales"] = act_scales
    logger.info("PTQ pipeline complete.")
    return model


# -----------------------------------------------------------------------------
# 11. Example: ImageNet (validation split!)
# -----------------------------------------------------------------------------

# %%
import torchvision.models as tvmodels
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Pretrained model -------------------------------------------------------
model_fp32 = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.DEFAULT)


# Datasets ---------------------------------------------------------------
def image_collate_fn(batch):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    images, labels = [], []
    for sample in batch:
        if isinstance(sample, dict):
            img = sample.get("img") or sample.get("image")
            label = sample.get("label", 0)
        else:
            img, label = sample
        img = transform(img) if isinstance(img, Image.Image) else transform(Image.fromarray(img))
        images.append(img)
        labels.append(label)
    return default_collate(images), default_collate(labels)


# Calibration subset (train split) --------------------------------------
if not Path(Config.Datasets.train_path).exists():
    logger.info("Downloading calibration subset …")
    calib_dsi = (
        load_dataset("imagenet-1k", split="train", streaming=True)
        .shuffle(seed=Config.seed, buffer_size=Config.Datasets.train_size)
        .take(Config.Datasets.train_size)
    )
    HFDataset.from_generator(lambda: (yield from calib_dsi), features=calib_dsi.features).save_to_disk(
        Config.Datasets.train_path
    )
calib_ds = HFDataset.load_from_disk(Config.Datasets.train_path)
calib_ds = calib_ds.select([i for i in range(len(calib_ds)) if calib_ds[i]["image"].mode != "L"])
calib_loader = DataLoader(calib_ds, batch_size=32, shuffle=True, collate_fn=image_collate_fn)

# **Validation** set for final accuracy ---------------------------------
if not Path(Config.Datasets.test_path).exists():
    logger.info("Downloading validation subset …")
    val_dsi = (
        load_dataset("imagenet-1k", split="validation", streaming=True)
        .shuffle(seed=Config.seed, buffer_size=Config.Datasets.test_size)
        .take(Config.Datasets.test_size)
    )
    HFDataset.from_generator(lambda: (yield from val_dsi), features=val_dsi.features).save_to_disk(
        Config.Datasets.test_path
    )
val_ds = HFDataset.load_from_disk(Config.Datasets.test_path)
val_ds = val_ds.select([i for i in range(len(val_ds)) if val_ds[i]["image"].mode != "L"])
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=image_collate_fn)

# Run PTQ ---------------------------------------------------------------
if not Path(Config.Quant.path).exists():
    logger.info("Generating quantised model …")
    q_model = ptq_pipeline(model_fp32, calib_loader)
    torch.save(q_model, Config.Quant.path)
else:
    q_model = torch.load(Config.Quant.path, weights_only=False)

# Evaluation ------------------------------------------------------------


def evaluate(model: nn.Module, dataloader, device: str | None = None, max_batches: int | None = None):
    model.eval()
    correct = total = 0
    device = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu")
    model.to(device)
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            if max_batches and i + 1 >= max_batches:
                break
    return correct / total if total else 0.0


acc_fp32 = evaluate(model_fp32, val_loader)
print(f"FP32 accuracy: {acc_fp32:.4%}")

acc_ptq = evaluate(q_model, val_loader)
print(f"PTQ accuracy: {acc_ptq:.4%}")
print(f"Accuracy drop: {(100 * (acc_fp32 - acc_ptq) / acc_fp32):.2f}%")
