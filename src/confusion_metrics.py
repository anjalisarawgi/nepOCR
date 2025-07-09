import pandas as pd
import difflib
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# === 1. Load your predictions CSV ===
df = pd.read_csv("results/evaluation_results_cleaned_bytebpe500_test_new8.csv")

# === 2. Count all (GT, Pred) character-level replacements ===
confusion_counter = Counter()

def extract_confusion_pairs(gt, pred):
    sm = difflib.SequenceMatcher(None, gt, pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            g = gt[i1:i2]
            p = pred[j1:j2]
            max_len = max(len(g), len(p))
            g = g.ljust(max_len)
            p = p.ljust(max_len)
            for gt_char, pred_char in zip(g, p):
                confusion_counter[(gt_char, pred_char)] += 1

for _, row in df.iterrows():
    extract_confusion_pairs(str(row["ground_truth"]), str(row["prediction"]))

# === 3. Convert to DataFrame ===
confusion_df = pd.DataFrame(
    [(gt, pred, count) for (gt, pred), count in confusion_counter.items()],
    columns=["ground_truth", "prediction", "count"]
)

confusion_df.to_csv("results/char_confusion_pairs.csv", index=False)

# === 4. Create pivot matrix ===
confusion_matrix = confusion_df.pivot_table(index="ground_truth", columns="prediction", values="count", fill_value=0)

# === 5. Focus on Top 30 Most Confused Characters ===
top_gt = confusion_df.groupby("ground_truth")["count"].sum().nlargest(30).index
top_pred = confusion_df.groupby("prediction")["count"].sum().nlargest(30).index
confusion_matrix_subset = confusion_matrix.loc[top_gt, top_pred]
confusion_matrix_subset = confusion_matrix_subset.astype(int)

# === 6. Load Devanagari Font ===
font_path = "fonts/NotoSansDevanagari-Regular.ttf"
devanagari_font = fm.FontProperties(fname=font_path)

# === 7. Plot heatmap ===
plt.figure(figsize=(12, 10))
ax = sns.heatmap(confusion_matrix_subset, annot=True, fmt="d", cmap="Reds", linewidths=0.5, linecolor='gray', cbar=True, square=True)

plt.title("🔤 Top OCR Character Confusions", fontsize=16, fontproperties=devanagari_font)
plt.xlabel("Predicted Character", fontproperties=devanagari_font)
plt.ylabel("Ground Truth Character", fontproperties=devanagari_font)

# Set tick fonts
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=devanagari_font, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=devanagari_font, rotation=0)

plt.tight_layout()
plt.savefig("results/char_confusion_heatmap_top30.png", dpi=300)
