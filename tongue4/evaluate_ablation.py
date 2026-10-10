import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from datasets.v7_dataset import V7TongueDataset
from models.strnet_final import STRNetFinal


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


MODELS = [

    "full",
    "no_temporal",
    "no_transformer",
    "no_channel",
    "no_dcrm",
    "no_reliability"

]


TEST_FILE = (
    "dataset_v7_continuous_synthetic/windows/"
    "window128_test_generatorA_ood.npz"
)


os.makedirs(
    "results",
    exist_ok=True
)



results = []


for name in MODELS:


    print("="*60)
    print("Testing:", name)


    model = STRNetFinal().to(DEVICE)


    model.load_state_dict(
        torch.load(
            f"ablation_checkpoints/{name}.pth",
            map_location=DEVICE
        )
    )


    model.eval()


    dataset = V7TongueDataset(
        TEST_FILE
    )


    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False
    )


    preds = []
    labels = []


    with torch.no_grad():

        for x, y, q in loader:

            x = x.to(DEVICE)


            output, _, _ = model(
                x,
                use_temporal=(name!="no_temporal"),
                use_transformer=(name!="no_transformer"),
                use_channel=(name!="no_channel"),
                use_reliability=(name!="no_reliability")
            )


            pred = output.argmax(
                dim=1
            ).cpu().numpy()


            preds.extend(pred)
            labels.extend(
                y.numpy()
            )


    acc = accuracy_score(
        labels,
        preds
    )


    f1 = f1_score(
        labels,
        preds,
        average="macro"
    )


    results.append(
        {
            "Model":name,
            "Accuracy":acc,
            "Macro_F1":f1
        }
    )


    print(
        f"{name}: "
        f"ACC={acc:.4f}, "
        f"F1={f1:.4f}"
    )



df = pd.DataFrame(results)


df.to_csv(
    "results/strnet_ablation_results.csv",
    index=False
)


print("\nAblation results:")
print(df)


print(
    "\nSaved:"
)

print(
    "results/strnet_ablation_results.csv"
)