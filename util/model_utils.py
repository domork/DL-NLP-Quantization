import torch
import torch.nn as nn
import torch.quantization

import util.process_data as process


def _print_header_footer(header, model_type, method):
    """
    Print header or footer for model processing.

    Args:
        header: True for header, False for footer
        model_type: Type of the model
        method: 'PTQ' or 'QAT'
    """
    if header:
        print(f"[*] started model processing with {method}")
        if method == "Quantization-Aware Training (QAT)":
            print(f"\n----- {model_type.upper()} with QAT -----\n")
        else:
            print(f"\n----- {model_type.upper()} -----\n")
    else:
        print(f"[+] {method.split(' ')[0]} Test complete")
        print(f"[+] finished model processing with {method.split(' ')[0]}")
        print("[*] ==========================================")


def _test_model(model_creator, model_name, device, test_loader, train_loader, model_type, quantize=False, fbgemm=False):
    """
    Test a model with the given parameters.

    Args:
        model_creator: Function to create the model
        model_name: Name of the saved model
        device: Device to run the model on
        test_loader: DataLoader for test data
        train_loader: DataLoader for training data
        model_type: Type of the model
        quantize: Whether to quantize the model
        fbgemm: Whether to use fbgemm backend for quantization
    """
    if quantize:
        print(f"[*] Testing quantized {model_type} model with {'PTQ' if fbgemm else 'QAT'}:")
        model = model_creator(num_classes=200, quantize=True)
        # Quantization operations must be done on CPU
        cpu_device = torch.device("cpu")
        test_device = cpu_device
    else:
        print(f"[*] Testing unquantized {model_type} model:")
        model = model_creator(num_classes=200)
        test_device = device

    loaded_dict_enc = torch.load(model_name, map_location=test_device)
    model.load_state_dict(loaded_dict_enc)

    if quantize and fbgemm:
        process.test(model=model, device=test_device, test_loader=test_loader, train_loader=train_loader,
                     quantize=True, fbgemm=True, model_type=model_type)
    else:
        process.test(model=model, device=test_device, test_loader=test_loader, train_loader=train_loader,
                     model_type=model_type)

    print(f"[+] Test complete")


def proceed_model(model_creator, model_type, device, epochs, train_loader, test_loader, lr, momentum,
                  quantize_param=False):
    """
    Process a model using Post-Training Quantization (PTQ).
    This function trains a model, tests it unquantized, and then applies PTQ.
    """
    method = "Post-Training Quantization (PTQ)"
    _print_header_footer(True, model_type, method)
    model_name = f"tiny_imagenet_{model_type}.pt"

    # Create and train the model on GPU if available
    model = model_creator(num_classes=200).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    print(f"[*] Training unquantized {model_type} model...")
    for epoch in range(1, epochs + 1):
        process.train(model, device, train_loader, optimizer, epoch)
    print(f"[+] Training complete")

    # Save the model
    torch.save(model.state_dict(), model_name)

    # Test unquantized model (can use GPU)
    _test_model(model_creator, model_name, device, test_loader, train_loader, model_type)

    # Test quantized model with Post-Training Quantization (PTQ)
    # _test_model will handle moving to CPU for quantization
    _test_model(model_creator, model_name, device, test_loader, train_loader, model_type, quantize=True, fbgemm=True)

    _print_header_footer(False, model_type, method)


def proceed_model_qat(model_creator, model_type, device, epochs, train_loader, test_loader, lr, momentum):
    """
    Process a model using Quantization-Aware Training (QAT).
    This function creates a quantization-aware model, trains it, and then converts it to a fully quantized model.
    """
    method = "Quantization-Aware Training (QAT)"
    _print_header_footer(True, model_type, method)
    model_name = f"tiny_imagenet_{model_type}_qat.pt"

    # Create a model with quantization enabled
    model = model_creator(num_classes=200, quantize=True)

    # Set model to evaluation mode for fusion
    model.eval()

    # Fuse modules (must be done on CPU)
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

    # Set QAT configuration
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Set model to training mode for QAT (required by prepare_qat)
    model.train()

    # Prepare model for QAT
    torch.quantization.prepare_qat(model, inplace=True)

    # Move to training device
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Train with QAT
    print(f"[*] Training {model_type} model with QAT...")
    for epoch in range(1, epochs + 1):
        process.train(model, device, train_loader, optimizer, epoch)
    print(f"[+] QAT Training complete")

    # Move back to CPU for quantization
    model.to('cpu')

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    # Save the QAT model
    torch.save(model.state_dict(), model_name)

    # Test the QAT model
    process.test(model=model, device=torch.device('cpu'), test_loader=test_loader, 
                 train_loader=train_loader, model_type=model_type)

    _print_header_footer(False, model_type, method)
