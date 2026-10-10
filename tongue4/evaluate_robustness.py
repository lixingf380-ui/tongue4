
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from datasets.v7_dataset import V7TongueDataset
from models.strnet import STRNetFinal


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "last_strnet_final.pth"

DATA_ROOT = "dataset_v7_continuous_synthetic/windows"


TEST_FILES = {
    "A_nominal": "window128_test_generatorA_nominal.npz",
    "A_low": "window128_test_generatorA_low.npz",
    "A_medium": "window128_test_generatorA_medium.npz",
    "A_high": "window128_test_generatorA_high.npz",
    "A_ood": "window128_test_generatorA_ood.npz",

    "B_nominal": "window128_test_generatorB_nominal.npz",
    "B_medium": "window128_test_generatorB_medium.npz",
    "B_high": "window128_test_generatorB_high.npz",
    "B_ood": "window128_test_generatorB_ood.npz",
}


def evaluate(model, file_path):

    dataset = V7TongueDataset(file_path)

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False
    )

    preds = []
    labels = []

    model.eval()

    with torch.no_grad():
        for x, y, q in loader:

            x = x.to(DEVICE)

            output, _, _ = model(x)

            pred = output.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)

    f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    return acc, f1


def main():

    print("=" * 60)
    print("ST-RNet Robustness Evaluation")
    print("Device:", DEVICE)
    print("=" * 60)


    model = STRNetFinal().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    results = []

    for name, file in TEST_FILES.items():

        path = os.path.join(
            DATA_ROOT,
            file
        )

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        print("\nTesting:", name)

        acc, f1 = evaluate(
            model,
            path
        )

        print(
            f"{name}: "
            f"Accuracy={acc:.4f}, "
            f"Macro-F1={f1:.4f}"
        )

        results.append(
            {
                "Dataset": name,
                "Accuracy": acc,
                "Macro-F1": f1
            }
        )


    df = pd.DataFrame(results)

    df.to_csv(
        "robustness_results.csv",
        index=False
    )

    print("\nSaved: robustness_results.csv")

    print(df)


if __name__ == "__main__":
    main()
