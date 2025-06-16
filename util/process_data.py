import os
import time

import torch
import torch.nn.functional as F

from util.model_fusing import get_modules_to_fuse


def train(model, device, train_loader, optimizer, epoch):
    """
    Train a model for one epoch.

    Args:
        model: The model to train
        device: Device to run training on (CPU or GPU)
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epoch: Current epoch number
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def test(model, device, test_loader, train_loader, quantize=False, fbgemm=False, model_type='resnet18'):
    """
    Test a model and print performance metrics.

    Args:
        model: The model to test
        device: Device to run testing on (CPU or GPU)
        test_loader: DataLoader for test data
        train_loader: DataLoader for training data (used for calibration in quantization)
        quantize: Whether to quantize the model before testing
        fbgemm: Whether to use fbgemm backend for quantization
        model_type: Type of the model ('resnet18', 'resnet50', or 'mobilenet_v2')
    """
    model.to(device)
    model.eval()

    if quantize:
        # Get modules to fuse based on model type
        modules_to_fuse = get_modules_to_fuse(model, model_type)

        model = torch.quantization.fuse_modules(model, modules_to_fuse)
        if fbgemm:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            model.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(model, inplace=True)
        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(train_loader):
                if i >= 100:
                    break
                data = data.to(device)
                model(data)
        torch.quantization.convert(model, inplace=True)

    test_loss = 0
    correct = 0
    inference_times = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            st = time.time()
            output = model(data)
            et = time.time()
            inference_times.append((et - st) * 1000)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("========================================= PERFORMANCE =============================================")
    print_size_of_model(model)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        print('Average inference time = {:0.4f} milliseconds'.format(avg_inference_time))
    print("====================================================================================================")
