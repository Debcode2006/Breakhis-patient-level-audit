"""Image transforms and augmentation.

Built on torchvision so the only heavy dependency is PyTorch itself. Augmentation
strength is fully config-driven; the defaults are histopathology-appropriate
(flips + mild rotation/jitter, no vertical-semantics assumptions) and conservative
enough to be stable on a small GPU.

ImageNet normalization statistics are used because the Swin backbone is
ImageNet-pretrained.
"""

from __future__ import annotations

from typing import Callable

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg, train: bool) -> Callable:
    """Build a torchvision transform pipeline from the augmentation config.

    Parameters
    ----------
    cfg:
        The full :class:`Config`; reads ``cfg.data.image_size`` and
        ``cfg.augmentation.*``.
    train:
        Whether to include stochastic augmentation. Validation / test use only
        resize + normalize so evaluation is deterministic.
    """
    image_size = cfg.data.image_size
    aug = cfg.augmentation

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    ops = [transforms.Resize((image_size, image_size))]

    if getattr(aug, "random_horizontal_flip", 0.0):
        ops.append(transforms.RandomHorizontalFlip(p=aug.random_horizontal_flip))
    if getattr(aug, "random_vertical_flip", 0.0):
        ops.append(transforms.RandomVerticalFlip(p=aug.random_vertical_flip))
    if getattr(aug, "random_rotation", 0):
        ops.append(transforms.RandomRotation(degrees=aug.random_rotation))
    if getattr(aug, "color_jitter", None):
        cj = aug.color_jitter
        ops.append(
            transforms.ColorJitter(
                brightness=getattr(cj, "brightness", 0.0),
                contrast=getattr(cj, "contrast", 0.0),
                saturation=getattr(cj, "saturation", 0.0),
                hue=getattr(cj, "hue", 0.0),
            )
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    if getattr(aug, "random_erasing", 0.0):
        ops.append(transforms.RandomErasing(p=aug.random_erasing))

    return transforms.Compose(ops)
