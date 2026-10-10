import os
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

DEVICE="cuda:1" if torch.cuda.is_available() else "cpu"

MODES = {
    "full": {},
    "no_temporal": {"use_temporal":False},
    "no_transformer": {"use_transformer":False},
    "no_channel": {"use_channel":False},
    "no_reliability": {"use_reliability":False}
}

DATA="dataset_v7_continuous_synthetic/windows/window128_train_generatorA_mixed.npz"


for name, cfg in MODES.items():

    print("Start", name)

    seed_everything(42)

    model = STRNetFinal().to(DEVICE)


    g = torch.Generator()
    g.manual_seed(42)

    g = torch.Generator()
    g.manual_seed(42)

    loader = DataLoader(
        V7TongueDataset(DATA),
        batch_size=512,
        shuffle=True,
        generator=g
    )

    opt=torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    loss_fn=nn.CrossEntropyLoss()

    for epoch in range(100):

        model.train()
        total=0

        for x,y,q in loader:

            x=x.to(DEVICE)
            y=y.to(DEVICE)

            opt.zero_grad()

            out,_,_=model(x,**cfg)

            loss=loss_fn(out,y)

            loss.backward()
            opt.step()

            total+=loss.item()

        print(
            name,
            epoch+1,
            total/len(loader)
        )

    os.makedirs("ablation_checkpoints",exist_ok=True)

    torch.save(
        model.state_dict(),
        f"ablation_checkpoints/{name}.pth"
    )
