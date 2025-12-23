import os
import yaml
import optuna
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data.dataloader import build_dataloaders
from src.models.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/processed"
BATCH_SIZE = 32
EPOCHS = 4   # short, on purpose

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 384])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.03, 0.12)
    lr = trial.suggest_float("lr_head", 5e-4, 2e-3, log=True)

    loaders = build_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = loaders["num_classes"]

    model = build_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone_flag=True,
        unfreeze_last=False,
        head_config={
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )

    for _ in range(EPOCHS):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loaders["val"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=12)

    best_params = study.best_params
    best_params["val_accuracy"] = study.best_value

    out_path = os.path.join(REPORT_DIR, "best_head_params.yaml")
    with open(out_path, "w") as f:
        yaml.dump(best_params, f)

    print("\n Best Head Params:")
    print(best_params)
    print(f" Saved to {out_path}")


if __name__ == "__main__":
    main()
