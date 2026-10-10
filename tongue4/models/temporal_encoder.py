import torch
import torch.nn as nn


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_channels=18, dim=128):
        super().__init__()

        self.multi_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, dim//4, 3, padding=1),
                nn.BatchNorm1d(dim//4),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, dim//4, 5, padding=2),
                nn.BatchNorm1d(dim//4),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, dim//4, 3, padding=2, dilation=2),
                nn.BatchNorm1d(dim//4),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, dim//4, 7, padding=3),
                nn.BatchNorm1d(dim//4),
                nn.GELU()
            )
        ])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        x = torch.cat([m(x) for m in self.multi_conv], dim=1)
        x = x.transpose(1,2)
        x = self.transformer(x)
        return x
