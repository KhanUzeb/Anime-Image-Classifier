import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.data.dataloader import build_dataloaders
from src.models.model import build_model

# =========================
# CONFIG
# =========================

DATA_DIR = "data/processed"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGE THIS PER RUN
#CHECKPOINT_PATH = "checkpoints/resnet34_frozen_best.pth"
CHECKPOINT_PATH = "checkpoints/resnet34_finetuned_best2.pth"
RUN_NAME = "finetuned_layer4"   # change to "finetuned_layer4"


REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# =========================
# EVALUATE
# =========================

def evaluate():
    print(f"\n Evaluating: {RUN_NAME}")
    print(f" Checkpoint: {CHECKPOINT_PATH}")

    # -------- Data --------
    loaders = build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=None,
    )

    test_loader = loaders["test"]
    class_names = list(loaders["class_to_idx"].keys())
    num_classes = loaders["num_classes"]

    # -------- Model --------
    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone_flag=True,
        unfreeze_last=False,  # irrelevant for eval
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    # -------- Inference --------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # -------- Metrics --------
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    df = pd.DataFrame(report).transpose()

    # Add run metadata
    df["run_name"] = RUN_NAME
    df["checkpoint"] = CHECKPOINT_PATH

    # -------- Save --------
    out_path = os.path.join(REPORT_DIR, f"{RUN_NAME}_report.csv")
    df.to_csv(out_path)

    print(f"📄 Report saved to: {out_path}")


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    evaluate()
