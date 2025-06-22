import torch
import torch.nn as nn

import util.model_utils as model_utils
from util.shared_resources import conv1x1


def mobilenet_v2(quantize=False, **kwargs):
    return MobileNetV2(quantize=quantize, **kwargs)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=200, width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8, norm_layer=None, quantize=False):
        super(MobileNetV2, self).__init__()
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0:
            raise ValueError("The inverted_residual_setting should be non-empty")

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        self.conv0 = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        self.bn0 = norm_layer(input_channel)
        self.relu0 = nn.ReLU6(inplace=True)

        self.features_blocks = nn.ModuleList()
        in_channels_block = input_channel
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features_blocks.append(InvertedResidual(in_channels_block, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, quantize=self.quantize))
                in_channels_block = output_channel

        self.conv_final = conv1x1(in_channels_block, self.last_channel)
        self.bn_final = norm_layer(self.last_channel)
        self.relu_final = nn.ReLU6(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.last_channel, num_classes)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        if self.quantize:
            x = self.quant(x)

        x = self.relu0(self.bn0(self.conv0(x)))

        for block in self.features_blocks:
            x = block(x)

        x = self.relu_final(self.bn_final(self.conv_final(x)))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.quantize:
            x = self.dequant(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, quantize=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1_pw = conv1x1(inp, hidden_dim)
        self.bn1_pw = norm_layer(hidden_dim)
        self.relu1_pw = nn.ReLU6(inplace=True)
        if expand_ratio == 1:
            self.conv_pw_sequential = nn.Identity()
        else:
            self.conv_pw_sequential = nn.Sequential(
                self.conv1_pw,
                self.bn1_pw,
                self.relu1_pw
            )

        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn_dw = norm_layer(hidden_dim)
        self.relu_dw = nn.ReLU6(inplace=True)

        self.conv_pw_linear = conv1x1(hidden_dim, oup)
        self.bn_pw_linear = norm_layer(oup)

        self.skip_add = nn.quantized.FloatFunctional()
        # Add DeQuantStub and QuantStub for handling quantized tensors
        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x):
        identity = x

        out = self.conv_pw_sequential(x)

        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = self.relu_dw(out)

        out = self.conv_pw_linear(out)
        out = self.bn_pw_linear(out)

        if self.use_res_connect:
            if self.quantize:
                # For quantized models, dequantize before addition and quantize after
                out = self.dequant(out)
                identity = self.dequant(identity)
                out += identity
                out = self.quant(out)
            else:
                out += identity

        return out



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    model_utils.proceed_model(mobilenet_v2, 'mobilenet_v2', device, epochs, train_loader, test_loader, lr, momentum, skip_training=skip_training)

def proceed_qat(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    model_utils.proceed_model_qat(mobilenet_v2, 'mobilenet_v2', device, epochs, train_loader, test_loader, lr, momentum, skip_training=skip_training)

def proceed(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=skip_training)
