import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, epoch):
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
    #
    # timestamp = str(time.time())
    # torch.save(model.state_dict(), timestamp)
    # print('Saved model at current timestamp: ' + timestamp)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def test(model, device, test_loader, train_loader, quantize=False, fbgemm=False, model_type='resnet18'):
    model.to(device)
    model.eval()

    if quantize:
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
