import os
import pandas as pd
import difflib
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


df = pd.read_csv("results/trocr_large_bert_byteBPE/predictions.csv")
font_path = "fonts/NotoSansDevanagari-Regular.ttf"
output_dir = "results/trocr_large_bert_byteBPE/"
os.makedirs(output_dir, exist_ok=True)

#  CER histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["cer"], bins=40, color="#1f77b4", edgecolor='black', alpha=0.75)
plt.title("Distribution of Character Error Rate (CER)", fontsize=16)
plt.xlabel("Character Error Rate", fontsize=14)
plt.ylabel("Number of Samples", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cer_histogram.png"), dpi=300)


# extracting confusion pairs 
confusions = Counter()

for _, row in df.iterrows():
    gt = str(row["ground_truth"])
    pred = str(row["prediction"])
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, pred).get_opcodes():
        if tag == 'replace':
            g = gt[i1:i2].ljust(max(i2 - i1, j2 - j1))
            p = pred[j1:j2].ljust(max(i2 - i1, j2 - j1))
            for gc, pc in zip(g, p):
                confusions[(gc, pc)] += 1
        # elif tag == 'insert':
        #     for pc in pred[j1:j2]:
        #         confusions[('', pc)] += 1  # insertion: nothing in GT, something in pred
        # elif tag == 'delete':
        #     for gc in gt[i1:i2]:
        #         confusions[(gc, '')] += 1  # deletion: something in GT and nothing in pred


# saving it as csv
df_conf = pd.DataFrame(
    [(g, p, c) for (g, p), c in confusions.items()],
    columns=["ground_truth", "prediction", "count"]
)
df_conf.to_csv(os.path.join(output_dir, "char_confusion_pairs.csv"), index=False)

pivot = df_conf.pivot_table(index="ground_truth", columns="prediction", values="count", fill_value=0)
top_gt = df_conf.groupby("ground_truth")["count"].sum().nlargest(30).index
top_pred = df_conf.groupby("prediction")["count"].sum().nlargest(30).index
matrix = pivot.loc[top_gt, top_pred].astype(int)

# heatmap for confusion pairs
font = fm.FontProperties(fname=font_path)
plt.figure(figsize=(12, 10))

ax = sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="OrRd", 
    linewidths=0.4,
    linecolor="gray",
    cbar=True,
    square=True,
    annot_kws={"fontsize": 10}
)

plt.title("Top 30 Most Frequent Character Confusions", fontsize=16, pad=12)
plt.xlabel("Predicted Character", fontsize=14)
plt.ylabel("Ground Truth Character", fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font, fontsize=12, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font, fontsize=12, rotation=0)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_pairs_heatmap.png"), dpi=300)
plt.close()



# mean and weighted cer
mean_cer = df["cer"].mean()
weighted_cer = (
    (df["cer"] * df["ground_truth"].str.len()).sum()  / df["ground_truth"].str.len().sum())

print("Mean cer:", mean_cer)
print("Weighted cer:", weighted_cer)