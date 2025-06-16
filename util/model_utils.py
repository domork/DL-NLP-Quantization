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