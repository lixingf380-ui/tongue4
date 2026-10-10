import torch
import torch.nn as nn


class ReliabilityGuidedFusion(nn.Module):
    def __init__(self, feature_dim=128, channels=6):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(feature_dim,64),
            nn.GELU(),
            nn.Linear(64,channels),
            nn.Sigmoid()
        )

    def forward(self,x):
        quality = self.predictor(x.mean(dim=1))

        weight = quality.mean(dim=1).view(-1,1,1)

        return x * weight, quality
