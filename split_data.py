"""
split_data.py

Splits data/raw into train/val/test (ImageFolder format)

- Assumes: data/raw/<class_name>/*.jpg
- Creates: data/processed/{train,val,test}/<class_name>/
"""

import shutil
import random
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def split_dataset():
    assert RAW_DIR.exists(), " data/raw does not exist"

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print("\n Splitting dataset...\n")

    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f" Class: {class_name}")

        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ]

        if len(images) < 5:
            print(f" Skipping {class_name} (too few images)")
            continue

        random.shuffle(images)

        total = len(images)
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in splits.items():
            out_class_dir = OUT_DIR / split / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy2(
                    img_path,
                    out_class_dir / img_path.name
                )

        print(
            f" {class_name}: "
            f"{len(splits['train'])} train | "
            f"{len(splits['val'])} val | "
            f"{len(splits['test'])} test"
        )

    print("\n Dataset split complete.")
    print(f" Output written to: {OUT_DIR}")


if __name__ == "__main__":
    split_dataset()
