import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from src.data.dataloader import build_dataloaders
from src.models.model import build_model
from src.utils import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "baseline": "checkpoints/resnet34_baseline_optuna.pth",
    "finetuned": "checkpoints/resnet34_finetuned_optuna.pth",
}

DEFAULT_PARAMS = {"hidden_dim": 256, "dropout": 0.3}


def get_uncertainty_metrics(model, loader):
    """Computes average confidence and entropy."""
    model.eval()
    confidences = []
    entropies = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            probs = F.softmax(model(images), dim=1)

            confidences.extend(probs.max(dim=1).values.cpu().tolist())
            
            # Entropy: -sum(p * log(p))
            batch_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropies.extend(batch_entropy.cpu().tolist())

    return (
        sum(confidences) / len(confidences),
        sum(entropies) / len(entropies)
    )


def get_gradient_norms(model, loader):
    """
    Computes gradient norms for Head vs Last Layer.
    Useful to see which part of the model is learning during fine-tuning.
    """
    model.train()
    model.zero_grad()
    
    images, labels = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    loss = F.cross_entropy(model(images), labels)
    loss.backward()

    head_grad = 0.0
    layer4_grad = 0.0

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.norm().item()
        if "fc" in name:
            head_grad += norm
        if "layer4" in name:
            layer4_grad += norm

    model.eval()
    return head_grad, layer4_grad



def evaluate_run(run_name, checkpoint_path, loaders, params, report_dir):
    print(f"\n{'-'*40}")
    print(f" EVALUATING: {run_name.upper()}")
    print(f" Cloned from: {checkpoint_path}")
    print(f"{'-'*40}")

    if not os.path.exists(checkpoint_path):
        print(f" [!] Checkpoint not found: {checkpoint_path}")
        return


    unfreeze_last = "finetuned" in run_name
    model = build_model(
        num_classes=loaders["num_classes"],
        pretrained=False,
        freeze_backbone_flag=True,
        unfreeze_last=unfreeze_last,
        head_config=params
    ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    print(" -> Generating predictions...")
    
    with torch.no_grad():
        for images, labels in tqdm(loaders["test"], desc="Test Set"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    
    acc = accuracy_score(y_true, y_pred)
    print(f" -> Test Accuracy: {acc * 100:.2f}%")

    report_dict = classification_report(
        y_true, y_pred, 
        target_names=loaders["class_to_idx"].keys(), 
        output_dict=True
    )
    df = pd.DataFrame(report_dict).transpose()
    df["run_name"] = run_name
    df["checkpoint"] = checkpoint_path
    
    out_file = os.path.join(report_dir, f"{run_name}_report.csv")
    df.to_csv(out_file)
    print(f" -> Report saved to: {out_file}")

    print(" -> Running Diagnostics...")
    val_conf, val_ent = get_uncertainty_metrics(model, loaders["val"])
    test_conf, test_ent = get_uncertainty_metrics(model, loaders["test"])

    print(f"    [Val]  Confidence: {val_conf:.3f} | Entropy: {val_ent:.3f}")
    print(f"    [Test] Confidence: {test_conf:.3f} | Entropy: {test_ent:.3f}")

    if unfreeze_last:
        hg, lg = get_gradient_norms(model, loaders["train"])
        print(f"    [Grads] Head: {hg:.4f} | Layer4: {lg:.4f}")
        dominance = "Backbone" if lg > hg else "Head"
        print(f"    [Info] {dominance} gradients check dominant.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.report_dir, exist_ok=True)


    params_path = "reports/best_head_params.yaml"
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = yaml.safe_load(f)
    else:
        params = DEFAULT_PARAMS

    print("Loading Data...")
    loaders = build_dataloaders(args.data_dir, args.batch_size)

    for name, path in CHECKPOINTS.items():
        evaluate_run(name, path, loaders, params, args.report_dir)

    print("\n[Done] All evaluations complete.")


if __name__ == "__main__":
    main()
