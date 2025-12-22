import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.dataloader import build_dataloaders
from src.models.model import build_model

# =========================
# CONFIG
# =========================

DATA_DIR = "data/processed"
BATCH_SIZE = 32          # start safe; try 48 later if VRAM allows
EPOCHS = 10               # frozen backbone converges fast
LR = 1e-3                # higher LR since only head is training
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# cuDNN speedups
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# =========================
# TRAIN
# =========================

def train():
    print(f"\n Training on: {DEVICE}")

    # -------- Data --------
    loaders = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=None,  # auto
    )

    num_classes = loaders["num_classes"]
    print(f" Classes: {num_classes}")

    # -------- Model --------
    model = build_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone_flag=True,  #  frozen backbone
        unfreeze_last=False,
        hidden_dim=512,
        dropout=0.4,
    ).to(DEVICE)

    # -------- Loss --------
    criterion = nn.CrossEntropyLoss()

    # -------- Optimizer (ONLY trainable params) --------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # -------- Scheduler --------
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_val_acc = 0.0

    # =========================
    # LOOP
    # =========================

    for epoch in range(1, EPOCHS + 1):
        print(f"\n Epoch {epoch}/{EPOCHS}")

        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(loaders["train"], desc="Train", leave=False)

        for images, labels in train_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.2f}%",
            )

        train_acc = 100 * correct / total
        print(f" Train Acc: {train_acc:.2f}%")

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in loaders["val"]:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f" Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # ---- Save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "resnet34_frozen_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f" Saved best model ({best_val_acc:.2f}%)")

    print("\n Training finished.")
    print(f" Best Val Acc: {best_val_acc:.2f}%")


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    train()
