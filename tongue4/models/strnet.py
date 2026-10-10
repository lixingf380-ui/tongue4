import torch.nn as nn

from .temporal_encoder import MultiScaleTemporalEncoder
from .channel_interaction import DynamicChannelRelation
from .reliability_module import ReliabilityGuidedFusion


class STRNetFinal(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.temporal = MultiScaleTemporalEncoder()
        self.channel = DynamicChannelRelation()
        self.reliability = ReliabilityGuidedFusion()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        x = self.temporal(x)

        x, relation = self.channel(x)

        x, quality = self.reliability(x)

        logits = self.classifier(x.transpose(1,2))

        return logits, quality, relation
