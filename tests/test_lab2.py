import torch
from PIL import Image

from models import AlexNet64, LeNet5, LeNetModern
from utils.dataset import get_dataloaders
from utils.transforms import get_test_transforms, get_train_transforms


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_lenet_architectures():
    inputs = torch.randn(2, 3, 64, 64)
    base = LeNet5()
    modern = LeNetModern()

    assert base(inputs).shape == (2, 10)
    assert modern(inputs).shape == (2, 10)
    assert _parameter_count(base) == 338_486
    assert _parameter_count(modern) == 446_006


def test_alexnet_normalization_variants():
    inputs = torch.randn(2, 3, 64, 64)
    for normalization in ("none", "lrn", "bn"):
        model = AlexNet64(norm_type=normalization)
        assert model(inputs).shape == (2, 10)


def test_transform_shapes():
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    assert get_train_transforms("cifar10", augment=True)(image).shape == (3, 64, 64)
    assert get_test_transforms("cifar10")(image).shape == (3, 64, 64)


def test_tiny_imagenet_uses_real_validation_split():
    train_loader, validation_loader, num_classes, _ = get_dataloaders(
        "tiny_imagenet",
        data_dir="./data",
        batch_size=8,
        num_workers=0,
        use_augmentation=False,
    )
    assert len(train_loader.dataset) == 100_000
    assert len(validation_loader.dataset) == 10_000
    assert num_classes == 200
