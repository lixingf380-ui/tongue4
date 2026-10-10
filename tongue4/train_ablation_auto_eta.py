import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.v7_dataset import V7TongueDataset
from models.strnet import STRNetFinal


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = (
    "dataset_v7_continuous_synthetic/windows/"
    "window128_train_generatorA_mixed.npz"
)

EPOCHS = 100

ABLATIONS = [
    "full",
    "no_dcrm",
    "no_rgfm",
    "no_mste",
    "no_transformer"
]


# 保持forward接口一致的替代模块
class IdentityChannel(nn.Module):
    def forward(self, x):
        return x, None


class IdentityReliability(nn.Module):
    def forward(self, x, *args):
        return x


class IdentityTemporal(nn.Module):
    def forward(self, x):
        return x


def modify_model(model, mode):

    if mode == "no_dcrm":
        model.channel = IdentityChannel()

    elif mode == "no_rgfm":
        model.reliability = IdentityReliability()

    elif mode == "no_mste":
        model.temporal = IdentityTemporal()

    elif mode == "no_transformer":
        if hasattr(model.temporal, "transformer"):
            model.temporal.transformer = nn.Identity()

    return model


def train_one(mode):

    print("\n" + "=" * 70)
    print("Start:", mode)
    print("=" * 70)

    dataset = V7TongueDataset(TRAIN_PATH)

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = STRNetFinal()
    model = modify_model(model, mode)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        epoch_start = time.time()
        model.train()
        total_loss = 0

        for x, y, q in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            result = model(x)

            if isinstance(result, tuple):
                output = result[0]
            else:
                output = result

            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start

        print(
            f"{mode} | "
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss={total_loss/len(loader):.4f} | "
            f"EpochTime={epoch_time:.1f}s"
        )

    os.makedirs(
        "ablation_checkpoints",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        f"ablation_checkpoints/{mode}.pth"
    )


if __name__ == "__main__":

    total_start = time.time()

    print("=" * 70)
    print("ST-RNet Ablation Experiment")
    print("Experiments:", len(ABLATIONS))
    print("Epochs each:", EPOCHS)
    print("Expected total experiments:", len(ABLATIONS))
    print("=" * 70)

    for idx, mode in enumerate(ABLATIONS):

        experiment_start = time.time()

        print(
            f"\n[{idx+1}/{len(ABLATIONS)}] "
            f"Running {mode}"
        )

        train_one(mode)

        used = time.time() - experiment_start

        finished = idx + 1
        remain_exp = len(ABLATIONS) - finished

        avg_time = (time.time() - total_start) / finished
        remain_time = avg_time * remain_exp

        print(
            "\nExperiment finished:",
            mode
        )

        print(
            f"Used: {used/60:.1f} min"
        )

        print(
            f"Average experiment time: "
            f"{avg_time/60:.1f} min"
        )

        print(
            f"Remaining experiments: "
            f"{remain_exp}"
        )

        print(
            f"Estimated remaining time: "
            f"{remain_time/3600:.2f} hours"
        )

        print(
            "Estimated finish timestamp:",
            time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(
                    time.time()+remain_time
                )
            )
        )


    total_time = time.time() - total_start

    print("\nAll ablation experiments completed.")
    print(
        f"Total time: {total_time/3600:.2f} hours"
    )
