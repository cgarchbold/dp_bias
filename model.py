import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18

# Define model
class MultiTaskResNet(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2, num_race_classes=7, pretrained = False, freeze_backbone=False):
        super(MultiTaskResNet, self).__init__()

        if pretrained:
            self.backbone = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)  # Load pretrained weights
        else:
            self.backbone = resnet18(weights=None)  # No pretrained weights

        self.backbone.fc = nn.Identity()  # Remove original classification head
        feature_dim = 512  # ResNet18 feature output size

        # Freeze the entire backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Three classification heads
        self.age_head = nn.Linear(feature_dim, num_age_classes)
        self.gender_head = nn.Linear(feature_dim, num_gender_classes)
        self.race_head = nn.Linear(feature_dim, num_race_classes)

    def forward(self, x):
        x = self.backbone(x)
        age_out = self.age_head(x)
        gender_out = self.gender_head(x)
        race_out = self.race_head(x)
        return age_out, gender_out, race_out
