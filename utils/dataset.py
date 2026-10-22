"""
ELEC 475: Lab 2 - Unified Dataset & DataLoader Factory
Supports CIFAR-10 (primary) and Tiny ImageNet-200 (secondary benchmark).
"""
import os
import urllib.request
import zipfile
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from .transforms import get_train_transforms, get_test_transforms
from .seed import seed_worker

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNetValidation(Dataset):
    """Tiny ImageNet validation split using ``val_annotations.txt`` labels."""

    def __init__(self, root: str, class_to_idx, transform=None):
        self.transform = transform
        images_dir = os.path.join(root, "images")
        annotations_path = os.path.join(root, "val_annotations.txt")

        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"Tiny ImageNet annotations not found: {annotations_path}")

        self.samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                filename, wnid = fields[0], fields[1]
                if wnid in class_to_idx:
                    self.samples.append((os.path.join(images_dir, filename), class_to_idx[wnid]))

        if not self.samples:
            raise RuntimeError(f"No labeled Tiny ImageNet validation images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir

def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64,
    seed: int = 42,
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")
        val_dir = os.path.join(tiny_dir, "val")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        val_set = TinyImageNetValidation(
            root=val_dir,
            class_to_idx=train_set.class_to_idx,
            transform=val_tf,
        )
        num_classes = 200
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader, num_classes, classes
