
import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs(
    "results",
    exist_ok=True
)


df = pd.read_csv(
    "ablation_results.csv"
)


plt.figure(
    figsize=(8,5)
)

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Ablation Study - Accuracy"
)

plt.tight_layout()

plt.savefig(
    "results/ablation_accuracy.png",
    dpi=300
)

plt.close()



plt.figure(
    figsize=(8,5)
)

plt.bar(
    df["Model"],
    df["Macro_F1"]
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylabel(
    "Macro-F1"
)

plt.title(
    "Ablation Study - Macro F1"
)

plt.tight_layout()

plt.savefig(
    "results/ablation_f1.png",
    dpi=300
)

plt.close()


print("Saved:")
print("results/ablation_accuracy.png")
print("results/ablation_f1.png")
