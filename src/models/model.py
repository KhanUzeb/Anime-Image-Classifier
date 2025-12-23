import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        x = F.normalize(x, dim=1)
        return self.fc2(x)


def build_model(
    num_classes,
    pretrained=True,
    freeze_backbone_flag=True,
    unfreeze_last=False,
    head_config=None
):
    if head_config is None:
        head_config = {"hidden_dim": 256, "dropout": 0.3}

    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    )

    if freeze_backbone_flag:
        for p in model.parameters():
            p.requires_grad = False

    model.fc = ClassificationHead( # type: ignore
        model.fc.in_features,
        num_classes,
        head_config["hidden_dim"],
        head_config["dropout"],
    )

    if unfreeze_last:
        for p in model.layer4.parameters():
            p.requires_grad = True

    return model
