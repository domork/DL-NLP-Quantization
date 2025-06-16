import torch
from models.resnet_generic import ResNet, BasicBlock
import util.model_utils as model_utils

def resnet18(quantize=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], quantize=quantize, **kwargs)

def proceed(device, epochs, train_loader, test_loader, lr, momentum):
    # Process model with Post-Training Quantization (PTQ)
    model_utils.proceed_model(resnet18, 'resnet18', device, epochs, train_loader, test_loader, lr, momentum)

    # Process model with Quantization-Aware Training (QAT)
    model_utils.proceed_model_qat(resnet18, 'resnet18', device, epochs, train_loader, test_loader, lr, momentum)
