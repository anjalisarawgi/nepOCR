import pandas as pd
import difflib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter
from torchmetrics.text import CharErrorRate

df = pd.read_csv("decoding/results/results_cleaned.csv", encoding="utf-8")

plt.figure(figsize=(10, 5))
plt.hist(df['cer'], bins=20, edgecolor='black')
plt.title("Distribution of Raw CER")
plt.xlabel("Character Error Rate")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("decoding/results/histogram.png")


mean_cer_all = df['cer'].mean()
median_cer_all = df['cer'].median()
print(f"Mean CER for all pages: {mean_cer_all:.4f}")
print(f"Median CER for all pages: {median_cer_all:.4f}")

# Corpus-level CER
cer = CharErrorRate()
corpus_level_cer = cer(df['pred_text'].astype(str).tolist(), df['ground_truth'].astype(str).tolist()).item()
print(f"Corpus-level CER (true total edits / total chars): {corpus_level_cer:.4f}")

# Perfect predictions
perfect_predictions = df[df['cer'] == 0]
print(f"Percentage of perfect predictions: {len(perfect_predictions) / len(df) * 100:.2f}%")

# Dotless CER
def strip_dots(text):
    return str(text).replace(".", "").replace("।", "")

gt_nodots = df['ground_truth'].apply(strip_dots).tolist()
pred_nodots = df['pred_text'].apply(strip_dots).tolist()
dotless_cer = cer(pred_nodots, gt_nodots).item()

print(f"CER with all dots removed from GT and Prediction: {dotless_cer:.4f}")
print(f"Δ CER (reduction): {corpus_level_cer - dotless_cer:.4f}")

# === 3. Character-level Confusion Matrix ===

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
    extract_confusion_pairs(str(row["ground_truth"]), str(row["pred_text"]))

# Save confusion data
confusion_df = pd.DataFrame(
    [(gt, pred, count) for (gt, pred), count in confusion_counter.items()],
    columns=["ground_truth", "pred_text", "count"]
)
confusion_df.to_csv("decoding/results/char_confusion_pairs.csv", index=False)

# Pivot matrix
confusion_matrix = confusion_df.pivot_table(index="ground_truth", columns="pred_text", values="count", fill_value=0)

# Top 30 most confused
top_gt = confusion_df.groupby("ground_truth")["count"].sum().nlargest(30).index
top_pred = confusion_df.groupby("pred_text")["count"].sum().nlargest(30).index
confusion_matrix_subset = confusion_matrix.loc[top_gt, top_pred].astype(int)

# === 4. Heatmap ===

# Load Devanagari font
font_path = "fonts/NotoSansDevanagari-Regular.ttf"
devanagari_font = fm.FontProperties(fname=font_path)

plt.figure(figsize=(12, 10))
ax = sns.heatmap(confusion_matrix_subset, annot=True, fmt="d", cmap="Reds", linewidths=0.5, linecolor='gray', cbar=True, square=True)

plt.title("🔤 Top OCR Character Confusions", fontsize=16, fontproperties=devanagari_font)
plt.xlabel("Predicted Character", fontproperties=devanagari_font)
plt.ylabel("Ground Truth Character", fontproperties=devanagari_font)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=devanagari_font, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=devanagari_font, rotation=0)

plt.tight_layout()
plt.savefig("decoding/results/heatmap.png", dpi=300)