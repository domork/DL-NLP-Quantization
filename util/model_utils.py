import torch
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


def _test_model(model_creator, model_name, device, test_loader, train_loader, model_type, quantize=False):
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
    """
    if quantize:
        print(f"[*] Testing unquantized {model_type} model with PTQ:")
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

    if quantize:
        process.test(model=model, device=test_device, test_loader=test_loader, train_loader=train_loader, quantize=True)
    else:
        process.test(model=model, device=test_device, test_loader=test_loader, train_loader=train_loader)

    print(f"[+] Test complete")


def proceed_model(model_creator, model_type, device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    method = "Post-Training Quantization (PTQ)"
    _print_header_footer(True, model_type, method)
    model_name = f"tiny_imagenet_{model_type}.pt"

    # Create the model on GPU if available
    model = model_creator(num_classes=200).to(device)

    if not skip_training:
        # Train the model
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        print(f"[*] Training unquantized {model_type} model...")
        for epoch in range(1, epochs + 1):
            process.train(model, device, train_loader, optimizer, epoch)
        print(f"[+] Training complete")

        # Save the model
        torch.save(model.state_dict(), model_name)
    else:
        print(f"[*] Skipping training for {model_type} model, attempting to load pre-trained model...")
        try:
            loaded_dict_enc = torch.load(model_name, map_location=device)
            model.load_state_dict(loaded_dict_enc)
            print(f"[+] Successfully loaded pre-trained model from {model_name}")
        except FileNotFoundError:
            print(f"[!] Warning: Pre-trained model {model_name} not found. Model will not be trained or loaded.")

    # Test unquantized model (can use GPU)
    _test_model(model_creator, model_name, device, test_loader, train_loader, model_type)

    # Test quantized model with Post-Training Quantization (PTQ)
    # _test_model will handle moving to CPU for quantization
    _test_model(model_creator, model_name, device, test_loader, train_loader, model_type, quantize=True)

    _print_header_footer(False, model_type, method)


def proceed_model_qat(model_creator, model_type, device, epochs, train_loader, test_loader, lr, momentum, skip_training=False):
    method = "Quantization-Aware Training (QAT)"
    _print_header_footer(True, model_type, method)
    model_name = f"tiny_imagenet_{model_type}_qat.pt"

    # Create the model and move to CPU for quantization preparation
    cpu_device = torch.device("cpu")
    model = model_creator(num_classes=200, quantize=True).to(cpu_device)

    # Prepare model for QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    if not skip_training:
        # Move model to training device
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        print(f"[*] Training {model_type} model with QAT...")
        for epoch in range(1, epochs + 1):
            process.train(model, device, train_loader, optimizer, epoch)
        print(f"[+] QAT Training complete")

        # Move back to CPU for quantization
        model = model.to(cpu_device)

        # Convert the trained model to a quantized model
        torch.quantization.convert(model, inplace=True)

        # Save the quantized model
        torch.save(model.state_dict(), model_name)
    else:
        print(f"[*] Skipping QAT training for {model_type} model, attempting to load pre-trained QAT model...")
        try:
            # Load the state dictionary
            loaded_dict_enc = torch.load(model_name, map_location=cpu_device)

            # First convert the model to a quantized model
            # This ensures the model structure matches the saved state dictionary
            torch.quantization.convert(model, inplace=True)

            # Now load the state dictionary
            model.load_state_dict(loaded_dict_enc)
            print(f"[+] Successfully loaded pre-trained QAT model from {model_name}")
        except FileNotFoundError:
            print(f"[!] Warning: Pre-trained QAT model {model_name} not found. Model will not be trained or loaded.")

    # Test the quantized model
    process.test(model=model, device=cpu_device, test_loader=test_loader, train_loader=train_loader)
    process.test(model=model, device=cpu_device, test_loader=test_loader, train_loader=train_loader, quantize=True)

    _print_header_footer(False, model_type, method)
