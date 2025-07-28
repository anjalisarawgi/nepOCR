import re
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text):
    txt = unicodedata.normalize("NFC", str(text))
    txt = CLEANUP.sub("", txt)
    return re.sub(r"\s+", "", txt)

df = pd.read_csv ("results/trocr_large_bert_byteBPE/predictions.csv")
df["gt_clean"]  = df["ground_truth"].apply(clean_text)
df["gt_length"] = df["gt_clean"].str.len()

bins = list(range(0, df["gt_length"].max() + 10, 10))
df["length_bin"] = pd.cut(df["gt_length"], bins, right=False)

binned = (
    df
    .groupby("length_bin")
    .apply(lambda g: pd.Series({
        "weighted_cer": np.average(g["cer"], weights=g["gt_length"]),
        "n_lines":     len(g)
    }))
    .reset_index()
)

# binned.to_csv("results/trocr_large_bert_byteBPE/weighted_cer_by_length.csv", index=False)
plt.figure(figsize=(10, 6))
plt.plot(binned["length_bin"].astype(str), binned["weighted_cer"], marker="o", linewidth=2)
plt.xlabel("Ground Truth Length (Bins)", fontsize=12, fontweight="bold")
plt.ylabel("Weighted CER", fontsize=12, fontweight="bold")
plt.title("Character Error Rate vs Line Length", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
# plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/trocr_large_bert_byteBPE/weighted_cer_by_length.png", dpi=300)
