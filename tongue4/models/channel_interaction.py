import torch
import torch.nn as nn


class DynamicChannelRelation(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.matmul(q, k.transpose(-1,-2))
        attn = torch.softmax(attn / (q.shape[-1]**0.5), dim=-1)

        out = torch.matmul(attn, v)
        return out, attn
