import torch
from models.resnet_generic import ResNet, BasicBlock
import util.model_utils as model_utils

def resnet18(quantize=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], quantize=quantize, **kwargs)

def proceed_ptq(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    model_utils.proceed_model(resnet18, 'resnet18', device, epochs, train_loader, test_loader, lr, momentum, skip_training=skip_training)

def proceed_qat(device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    model_utils.proceed_model_qat(resnet18, 'resnet18', device, epochs, train_loader, test_loader, lr, momentum, skip_training=skip_training)
