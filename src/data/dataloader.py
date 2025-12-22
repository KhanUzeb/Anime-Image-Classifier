import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)

# =========================
# MAIN API
# =========================

def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int | None = None,
):
    """
    Builds fast, GPU-safe dataloaders.
    """

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)

    train_ds = ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_train_transforms(),
    )

    val_ds = ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=get_eval_transforms(),
    )

    test_ds = ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,   # IMPORTANT for BatchNorm
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": len(train_ds.classes),
        "class_to_idx": train_ds.class_to_idx,
    }

