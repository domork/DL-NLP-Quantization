import torch
from models.resnet_generic import ResNet, Bottleneck
import util.model_utils as model_utils


def resnet50(quantize=False, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], quantize=quantize, **kwargs)

def proceed(device, epochs, train_loader, test_loader, lr, momentum):
    model_utils.proceed_model(resnet50, 'resnet50', device, epochs, train_loader, test_loader, lr, momentum)
