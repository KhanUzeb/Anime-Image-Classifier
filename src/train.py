import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataloader import build_dataloaders
from src.models.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/processed"
PARAMS_PATH = "reports/best_head_params.yaml"
CHECKPOINT = "checkpoints/resnet34_baseline_optuna.pth"


def train():
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    loaders = build_dataloaders(DATA_DIR, 32)
    num_classes = loaders["num_classes"]

    model = build_model(
        num_classes,
        pretrained=True,
        freeze_backbone_flag=True,
        head_config=params
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=params["label_smoothing"]
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=params["lr_head"],
        weight_decay=1e-4
    )

    best_val = 0
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_loader = loaders["train"]
        
        # Training loop with tqdm
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch") as pbar:
            for x, y in pbar:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        
        # Validation loop with tqdm
        val_loader = loaders["val"]
        with tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", unit="batch") as pbar:
            with torch.no_grad():
                for x, y in pbar:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

        acc = correct / total
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {acc:.4f}")

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  -> Model saved! New best accuracy: {best_val:.4f}")

    print(f"\nBest baseline acc: {best_val:.3f}")


if __name__ == "__main__":
    train()
