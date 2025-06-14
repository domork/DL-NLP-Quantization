import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import zipfile
import urllib.request

def prepare_cuda(seed = 1):
    print("[*] preparing CUDA")
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise Exception("cuda could not be initialized")
    torch.manual_seed(seed)
    print("[+] CUDA is provided")
    return torch.device("cuda" if use_cuda else "cpu")

def download_and_extract_tiny_imagenet(data_dir='tiny-imagenet-200'):
    print("[*] checking is dataset provided...")
    url = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = 'tiny-imagenet-200.zip'

    if not os.path.exists(data_dir):
        print("[*] Is not provided. Downloading...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
        print("[+] Downloading complete")
    print("[+] finished dataset check ")

def check_val_data(val_dir='tiny-imagenet-200/val'):
    if not os.path.exists(val_dir):
        return False
    val_img_dir = os.path.join(val_dir, 'images')
    val_anno_file = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.exists(val_img_dir) or not os.path.exists(val_anno_file):
        return False
    return True

def get_tiny_imagenet_dataloaders():
    print("[*] started getting tiny imagenet dataloaders")
    data_dir = 'tiny-imagenet-200'
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
    print("[+] finished getting tiny imagenet dataloaders")
    return train_loader, test_loader
