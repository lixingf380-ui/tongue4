import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.v7_dataset import V7TongueDataset
from models.strnet_final import STRNetFinal
import random
import numpy as np


def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

TRAIN_FILE = "dataset_v7_continuous_synthetic/windows/window128_train_generatorA_mixed.npz"
VAL_FILE = "dataset_v7_continuous_synthetic/windows/window128_validation_generatorA_mixed.npz"

EPOCHS = 100
BATCH = 512


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_fn = nn.CrossEntropyLoss()
    total = 0

    with torch.set_grad_enabled(train):
        for x, y, q in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            if train:
                optimizer.zero_grad()

            out, _, _ = model(x)
            loss = loss_fn(out, y)

            if train:
                loss.backward()
                optimizer.step()

            total += loss.item()

    return total / len(loader)


g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    V7TongueDataset(TRAIN_FILE),
    batch_size=BATCH,
    shuffle=True,
    generator=g
)

val_loader = DataLoader(
    V7TongueDataset(VAL_FILE),
    batch_size=BATCH
)

model = STRNetFinal().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

os.makedirs("checkpoints", exist_ok=True)

best = 1e9
start = time.time()

for epoch in range(EPOCHS):

    train_loss = run_epoch(model, train_loader, optimizer)
    val_loss = run_epoch(model, val_loader)

    remain = (time.time()-start)/(epoch+1)*(EPOCHS-epoch-1)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"train={train_loss:.5f} "
        f"val={val_loss:.5f} "
        f"ETA={remain/60:.1f}min"
    )

    if val_loss < best:
        best = val_loss
        torch.save(
            model.state_dict(),
            "checkpoints/strnet_final_best.pth"
        )

print("finished")
