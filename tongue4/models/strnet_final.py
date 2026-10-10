
import torch
import torch.nn as nn


class STRNetFinal(nn.Module):
    """
    Final ST-RNet for tongue pressure dataset.

    Dataset output:
        (B,18,128)

    Supports ablation:
        use_temporal
        use_transformer
        use_channel
        use_reliability
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Keep dimension stable for all ablation settings
        self.input_projection = nn.Conv1d(
            18, 128, kernel_size=1
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(
                128, 128,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.channel_q = nn.Linear(128,128)
        self.channel_k = nn.Linear(128,128)
        self.channel_v = nn.Linear(128,128)

        self.reliability_head = nn.Sequential(
            nn.Linear(128,64),
            nn.GELU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128,num_classes)
        )


    def forward(
        self,
        x,
        use_temporal=True,
        use_transformer=True,
        use_channel=True,
        use_reliability=True
    ):

        # x:
        # B,18,128

        x = self.input_projection(x)

        # B,128,128

        if use_temporal:
            x = self.temporal(x)

        # Transformer expects:
        # B,T,C

        x = x.transpose(1,2)

        if use_transformer:
            x = self.transformer(x)


        relation = None

        if use_channel:

            q = self.channel_q(x)
            k = self.channel_k(x)
            v = self.channel_v(x)

            relation = torch.softmax(
                torch.matmul(
                    q,
                    k.transpose(-1,-2)
                ) / (128 ** 0.5),
                dim=-1
            )

            x = torch.matmul(
                relation,
                v
            )


        reliability = None

        if use_reliability:

            reliability = self.reliability_head(
                x.mean(dim=1)
            )

            x = x * reliability.unsqueeze(1)


        logits = self.classifier(
            x.transpose(1,2)
        )

        return logits, reliability, relation
