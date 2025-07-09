import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# === Step 1: Load your confusion data ===
df = pd.read_csv("results/char_confusion_pairs.csv")
font_path = "fonts/NotoSansDevanagari-Regular.ttf"


# === Step 2: Get top 10 most frequent confusions ===
top_10 = df.sort_values(by="count", ascending=False).head(10)

# === Step 3: Categorize Confusions ===
categories = {
    "halant_issues": ["्", "ं", "ः", "ँ"],
    "vowel_matras": ["ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ृ", "ॉ", "ॆ", "ॅ"],
    "missing_space": [" ", ""],
    "punctuation": ["।", "॥", ".", ","],
    "digits": [str(i) for i in range(10)],
}

def categorize(row):
    gt, pred = row["ground_truth"], row["prediction"]
    for cat, chars in categories.items():
        if gt in chars or pred in chars:
            return cat
    return "other"

df["category"] = df.apply(categorize, axis=1)
category_summary = df.groupby("category")["count"].sum().sort_values(ascending=False)

print("Confusion Counts by Category:")
print(category_summary)

# === Step 4: Plot Heatmap for Top 10 Confusions ===
pivot_df = top_10.pivot(index="ground_truth", columns="prediction", values="count").fillna(0)
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".0f", linewidths=0.5, cbar=True)
plt.title("Top 10 Most Frequent OCR Confusions")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("results/top_10_confusions_heatmap.png")