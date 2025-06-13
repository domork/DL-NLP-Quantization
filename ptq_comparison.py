import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import zipfile
import urllib.request

def download_and_extract_tiny_imagenet(data_dir='./tiny-imagenet-200'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = 'tiny-imagenet-200.zip'

    if not os.path.exists(data_dir):
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)

def check_val_data(val_dir='./tiny-imagenet-200/val'):
    if not os.path.exists(val_dir):
        return False
    val_img_dir = os.path.join(val_dir, 'images')
    val_anno_file = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(val_img_dir) or not os.path.exists(val_anno_file):
        return False
    return True

def get_tiny_imagenet_dataloaders():
    data_dir = './tiny-imagenet-200'
    download_and_extract_tiny_imagenet(data_dir)

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)

    val_dir = os.path.join(data_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_anno_file = os.path.join(val_dir, 'val_annotations.txt')
    val_target_dir = os.path.join(val_dir, 'organized')

    if not check_val_data(val_dir):
        raise FileNotFoundError(f"Validation data not found at {val_dir}")

    if not os.path.exists(val_target_dir):
        os.makedirs(val_target_dir)
        with open(val_anno_file) as f:
            for line in f:
                fname, class_name = line.split('\t')[:2]
                class_folder = os.path.join(val_target_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)
                src_path = os.path.join(val_img_dir, fname)
                dst_path = os.path.join(class_folder, fname)
                if os.path.exists(src_path):
                    os.rename(src_path, dst_path)

    test_dataset = torchvision.datasets.ImageFolder(val_target_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, quantize=False):
        super(BasicBlock, self).__init__()
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.quantize:
            out = self.skip_add.add(out, identity)
        else:
            out += identity

        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, quantize=False):
        super(Bottleneck, self).__init__()
        self.quantize = quantize
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.quantize:
            out = self.skip_add.add(out, identity)
        else:
            out += identity

        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=200, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, quantize=False):
        super(ResNet, self).__init__()
        self.quantize = quantize
        num_channels = 3
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, quantize=self.quantize))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, quantize=self.quantize))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.quantize:
            x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.quantize:
            x = self.dequant(x)
        
        return x

    def forward(self, x):
        return self._forward_impl(x)

def resnet18(quantize=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], quantize=quantize, **kwargs)

def resnet50(quantize=False, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], quantize=quantize, **kwargs)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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
                out = self.skip_add.add(out, identity)
            else:
                out += identity
        
        return out

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

def mobilenet_v2(quantize=False, **kwargs):
    return MobileNetV2(quantize=quantize, **kwargs)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
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
            modules_to_fuse = []
            modules_to_fuse.append(['conv0', 'bn0'])

            for i, block in enumerate(model.features_blocks):
                if isinstance(block.conv_pw_sequential, nn.Sequential):
                    modules_to_fuse.append([f'features_blocks.{i}.conv_pw_sequential.0', f'features_blocks.{i}.conv_pw_sequential.1'])
                
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

def main():
    batch_size = 64
    epochs = 20
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    save_model = True
    no_cuda = False
    
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    try:
        train_loader, test_loader = get_tiny_imagenet_dataloaders()
    except FileNotFoundError as e:
        print(e)
        return
    
    print("\n----- ResNet18 -----\n")
    model_type = 'resnet18'
    model_name = f"tiny_imagenet_{model_type}.pt"

    model_resnet18 = resnet18(num_classes=200).to(device)
    optimizer_resnet18 = torch.optim.SGD(model_resnet18.parameters(), lr=lr, momentum=momentum)
    args = {"log_interval": log_interval}
    
    print(f"Training unquantized {model_type} model...")
    for epoch in range(1, epochs + 1):
        train(args, model_resnet18, device, train_loader, optimizer_resnet18, epoch)
    
    if save_model:
        torch.save(model_resnet18.state_dict(), model_name)
    
    print(f"\nTesting unquantized {model_type} model:")
    cpu_device = 'cpu'
    encoder_resnet18 = resnet18(num_classes=200)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_resnet18.load_state_dict(loaded_dict_enc)
        test(model=encoder_resnet18, device=cpu_device, test_loader=test_loader, train_loader=train_loader, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")
    
    print(f"\nTesting quantized {model_type} model:")
    encoder_resnet18_quant = resnet18(num_classes=200, quantize=True)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_resnet18_quant.load_state_dict(loaded_dict_enc)
        test(model=encoder_resnet18_quant, device=cpu_device, test_loader=test_loader, train_loader=train_loader, quantize=True, fbgemm=True, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")

    print("\n----- ResNet50 -----\n")
    model_type = 'resnet50'
    model_name = f"tiny_imagenet_{model_type}.pt"

    model_resnet50 = resnet50(num_classes=200).to(device)
    optimizer_resnet50 = torch.optim.SGD(model_resnet50.parameters(), lr=lr, momentum=momentum)

    print(f"Training unquantized {model_type} model...")
    for epoch in range(1, epochs + 1):
        train(args, model_resnet50, device, train_loader, optimizer_resnet50, epoch)

    if save_model:
        torch.save(model_resnet50.state_dict(), model_name)

    print(f"\nTesting unquantized {model_type} model:")
    encoder_resnet50 = resnet50(num_classes=200)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_resnet50.load_state_dict(loaded_dict_enc)
        test(model=encoder_resnet50, device=cpu_device, test_loader=test_loader, train_loader=train_loader, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")

    print(f"\nTesting quantized {model_type} model:")
    encoder_resnet50_quant = resnet50(num_classes=200, quantize=True)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_resnet50_quant.load_state_dict(loaded_dict_enc)
        test(model=encoder_resnet50_quant, device=cpu_device, test_loader=test_loader, train_loader=train_loader, quantize=True, fbgemm=True, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")

    print("\n----- MobileNetV2 -----\n")
    model_type = 'mobilenet_v2'
    model_name = f"tiny_imagenet_{model_type}.pt"

    model_mobilenet_v2 = mobilenet_v2(num_classes=200).to(device)
    optimizer_mobilenet_v2 = torch.optim.SGD(model_mobilenet_v2.parameters(), lr=lr, momentum=momentum)

    print(f"Training unquantized {model_type} model...")
    for epoch in range(1, epochs + 1):
        train(args, model_mobilenet_v2, device, train_loader, optimizer_mobilenet_v2, epoch)

    if save_model:
        torch.save(model_mobilenet_v2.state_dict(), model_name)

    print(f"\nTesting unquantized {model_type} model:")
    encoder_mobilenet_v2 = mobilenet_v2(num_classes=200)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_mobilenet_v2.load_state_dict(loaded_dict_enc)
        test(model=encoder_mobilenet_v2, device=cpu_device, test_loader=test_loader, train_loader=train_loader, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")

    print(f"\nTesting quantized {model_type} model:")
    encoder_mobilenet_v2_quant = mobilenet_v2(num_classes=200, quantize=True)
    try:
        loaded_dict_enc = torch.load(model_name, map_location=cpu_device)
        encoder_mobilenet_v2_quant.load_state_dict(loaded_dict_enc)
        test(model=encoder_mobilenet_v2_quant, device=cpu_device, test_loader=test_loader, train_loader=train_loader, quantize=True, fbgemm=True, model_type=model_type)
    except FileNotFoundError:
        print(f"Trained model file '{model_name}' not found!")


if __name__ == '__main__':
    main()