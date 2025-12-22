import torch.nn as nn
from torchvision import models


# =========================
# CLASSIFIER HEAD
# =========================

class ClassificationHead(nn.Module):
    """
    Simple but strong MLP head.
    Designed for transfer learning.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# BACKBONE UTILS
# =========================

def freeze_backbone(model: nn.Module):
    """Freeze all backbone layers."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_block(model: nn.Module):
    """
    Unfreeze only the last ResNet block (layer4).
    Used for controlled fine-tuning.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True


# =========================
# MODEL FACTORY
# =========================

def build_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone_flag: bool = True,
    unfreeze_last: bool = False,
    hidden_dim: int = 512,
    dropout: float = 0.4,
):
    """
    Builds a ResNet34 model with a custom classification head.

    Args:
        num_classes: number of output classes
        pretrained: load ImageNet weights
        freeze_backbone_flag: freeze full backbone (Phase 1)
        unfreeze_last: unfreeze layer4 only (Phase 3)
        hidden_dim: classifier hidden size
        dropout: dropout in head

    Returns:
        nn.Module
    """

    # -------- Load backbone --------
    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # -------- Freeze backbone --------
    if freeze_backbone_flag:
        freeze_backbone(model)

    # -------- Get feature size --------
    in_features = model.fc.in_features

    # -------- Replace head --------
    model.fc = ClassificationHead(  # type: ignore
        in_features=in_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # -------- Partial unfreeze --------
    if unfreeze_last:
        unfreeze_last_block(model)

    return model
