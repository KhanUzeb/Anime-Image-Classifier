import os
import random
import numpy as np
import torch


# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# METRICS
# =========================

def accuracy_from_logits(logits, labels):
    """
    Computes accuracy from raw model outputs.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


# =========================
# CHECKPOINTING
# =========================

def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


# =========================
# PARAM COUNT (DEBUG / LOG)
# =========================

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())
