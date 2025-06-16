import torch
import torch.nn as nn

def get_modules_to_fuse(model, model_type):
    """
    Generate a list of modules to fuse for quantization based on model type.

    Args:
        model: The model instance
        model_type: Type of the model ('resnet18', 'resnet50', or 'mobilenet_v2')

    Returns:
        List of module pairs to fuse (each pair is a list of two module names)

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == 'resnet18':
        modules_to_fuse = [['conv1', 'bn1'],
                           ['layer1.0.conv1', 'layer1.0.bn1'],
                           ['layer1.0.conv2', 'layer1.0.bn2'],
                           ['layer1.1.conv1', 'layer1.1.bn1'],
                           ['layer1.1.conv2', 'layer1.1.bn2'],
                           ['layer2.0.conv1', 'layer2.0.bn1'],
                           ['layer2.0.conv2', 'layer2.0.bn2'],
                           ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
                           ['layer2.1.conv1', 'layer2.1.bn1'],
                           ['layer2.1.conv2', 'layer2.1.bn2'],
                           ['layer3.0.conv1', 'layer3.0.bn1'],
                           ['layer3.0.conv2', 'layer3.0.bn2'],
                           ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
                           ['layer3.1.conv1', 'layer3.1.bn1'],
                           ['layer3.1.conv2', 'layer3.1.bn2'],
                           ['layer4.0.conv1', 'layer4.0.bn1'],
                           ['layer4.0.conv2', 'layer4.0.bn2'],
                           ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
                           ['layer4.1.conv1', 'layer4.1.bn1'],
                           ['layer4.1.conv2', 'layer4.1.bn2']]
    elif model_type == 'resnet50':
        modules_to_fuse = [['conv1', 'bn1']]
        for i in range(1, 5):
            layer = getattr(model, f'layer{i}')
            for j in range(len(layer)):
                block = getattr(layer, str(j))
                modules_to_fuse.append([f'layer{i}.{j}.conv1', f'layer{i}.{j}.bn1'])
                modules_to_fuse.append([f'layer{i}.{j}.conv2', f'layer{i}.{j}.bn2'])
                modules_to_fuse.append([f'layer{i}.{j}.conv3', f'layer{i}.{j}.bn3'])
                if block.downsample:
                    modules_to_fuse.append([f'layer{i}.{j}.downsample.0', f'layer{i}.{j}.downsample.1'])
    elif model_type == 'mobilenet_v2':
        modules_to_fuse = [['conv0', 'bn0']]
        for i, block in enumerate(model.features_blocks):
            if isinstance(block.conv_pw_sequential, nn.Sequential):
                modules_to_fuse.append(
                    [f'features_blocks.{i}.conv_pw_sequential.0', f'features_blocks.{i}.conv_pw_sequential.1'])
            modules_to_fuse.append([f'features_blocks.{i}.conv_dw', f'features_blocks.{i}.bn_dw'])
            modules_to_fuse.append([f'features_blocks.{i}.conv_pw_linear', f'features_blocks.{i}.bn_pw_linear'])
        modules_to_fuse.append(['conv_final', 'bn_final'])
    else:
        raise ValueError("Unsupported model_type for quantization fusing.")

    return modules_to_fuse