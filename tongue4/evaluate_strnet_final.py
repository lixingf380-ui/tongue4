import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from datasets.v7_dataset import V7TongueDataset
from models.strnet_final import STRNetFinal


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "checkpoints/strnet_final_best.pth"


TEST_FILES = {

    "A_nominal":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorA_nominal.npz",

    "A_high":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorA_high.npz",

    "A_medium":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorA_medium.npz",

    "A_low":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorA_low.npz",

    "A_ood":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorA_ood.npz",

    "B_nominal":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorB_nominal.npz",

    "B_high":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorB_high.npz",

    "B_medium":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorB_medium.npz",

    "B_ood":
    "dataset_v7_continuous_synthetic/windows/window128_test_generatorB_ood.npz",
}


os.makedirs(
    "results",
    exist_ok=True
)


# load model

model = STRNetFinal().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


results = []


for name, file in TEST_FILES.items():

    print("="*60)
    print("Testing:", name)


    dataset = V7TongueDataset(file)


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

            output, _, _ = model(x)

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


    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )


    results.append(
        {
            "Dataset": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "Macro_F1": f1
        }
    )


    print(
        f"{name}: "
        f"ACC={acc:.4f}, "
        f"Macro-F1={f1:.4f}"
    )


    # save nominal confusion matrix

    if name == "A_nominal":

        cm = confusion_matrix(
            labels,
            preds
        )

        np.savetxt(
            "results/strnet_final_confusion_matrix.csv",
            cm,
            fmt="%d"
        )


df = pd.DataFrame(results)


df.to_csv(
    "results/strnet_final_test_report.csv",
    index=False
)


print("\nFinal results:")
print(df)


print(
    "\nSaved:"
)

print(
    "results/strnet_final_test_report.csv"
)

print(
    "results/strnet_final_confusion_matrix.csv"
)