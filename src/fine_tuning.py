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
CHECKPOINT_IN = "checkpoints/resnet34_frozen_best.pth"
CHECKPOINT_OUT = "checkpoints/resnet34_finetuned_best2.pth"

BATCH_SIZE = 32
EPOCHS = 8                  # fine-tuning converges fast
LR = 1e-4                   #  lower LR for backbone
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("checkpoints", exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# =========================
# FINE-TUNE
# =========================

def fine_tune():
    print(f"\n Fine-tuning on: {DEVICE}")

    # -------- Data --------
    loaders = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=None,
    )

    num_classes = loaders["num_classes"]
    print(f" Classes: {num_classes}")

    # -------- Model --------
    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone_flag=True,   # freeze all first
        unfreeze_last=True,          #  unfreeze layer4
        hidden_dim=512,
        dropout=0.4,
    ).to(DEVICE)

    # Load frozen-backbone weights
    state = torch.load(CHECKPOINT_IN, map_location=DEVICE)
    model.load_state_dict(state)

    # -------- Loss --------
    criterion = nn.CrossEntropyLoss()

    # -------- Optimizer (ONLY trainable params) --------
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_val_acc = 0.0

    # =========================
    # LOOP
    # =========================

    for epoch in range(1, EPOCHS + 1):
        print(f"\n Fine-Tune Epoch {epoch}/{EPOCHS}")

        # ---- Train ----
        model.train()
        correct, total = 0, 0

        train_bar = tqdm(loaders["train"], desc="Train", leave=False)

        for images, labels in train_bar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in loaders["val"]:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Val Acc: {val_acc:.2f}%")
        scheduler.step()

        # ---- Save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_OUT)
            print(f" Saved best fine-tuned model ({best_val_acc:.2f}%)")

    print("\n Fine-tuning finished.")
    print(f" Best Val Acc: {best_val_acc:.2f}%")


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    fine_tune()
