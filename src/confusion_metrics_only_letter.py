import pandas as pd
import difflib
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re

# ─── CONFIG ─────────────────────────────────────────────────────────────
EVAL_CSV       = "results/evaluation_results_cleaned_bytebpe500_test_new8.csv"
FONT_PATH      = "fonts/NotoSansDevanagari-Regular.ttf"
OUTPUT_PNG     = "results/char_confusion_heatmap_top30_v2.png"
TOP_K          = 30        # how many GT / Pred characters to display
IGNORE_MATRAS  = True      # whether to drop matras/punctuation/spaces
CSV_OUT = "results/char_confusion_matrix_top30_v2.csv"

# ────────────────────────────────────────────────────────────────────────

# Devanagari matras & common non-letter chars
MATRAS = set(" ा ि ी ु ू ृ े ै ो ौ ं ः ँ । ॥".split())
PAD    = " "  # how we’ll pad mismatched spans

# simple helper to test “real” Devanagari letters (no matras/pad)
def is_dev_letter(c: str) -> bool:
    return bool(re.match(r'[\u0900-\u097F]', c)) and c not in MATRAS and c != PAD

# ─── 1. Load predictions ─────────────────────────────────────────────────
df = pd.read_csv(EVAL_CSV, dtype=str).fillna("")

# ─── 2. Extract all char-level confusions ────────────────────────────────
counter = Counter()

def record_confusions(gt: str, pred: str):
    sm = difflib.SequenceMatcher(None, gt, pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert", "delete"):
            g_chunk = gt[i1:i2]
            p_chunk = pred[j1:j2]
            # pad shorter to equal length so zip() pairs
            maxlen = max(len(g_chunk), len(p_chunk))
            g_chunk = g_chunk.ljust(maxlen, PAD)
            p_chunk = p_chunk.ljust(maxlen, PAD)
            for g, p in zip(g_chunk, p_chunk):
                # optionally ignore pad/matras
                if IGNORE_MATRAS and (g in MATRAS or p in MATRAS or g == PAD or p == PAD):
                    continue
                counter[(g, p)] += 1

for _, row in df.iterrows():
    record_confusions(row["ground_truth"], row["prediction"])

# ─── 3. Build a flat DataFrame ───────────────────────────────────────────
flat = [
    (g, p, cnt)
    for (g, p), cnt in counter.items()
]
conf_df = pd.DataFrame(flat, columns=["ground_truth", "prediction", "count"])

# ─── 4. Pivot into a matrix ──────────────────────────────────────────────
cm = conf_df.pivot_table(
    index="ground_truth", columns="prediction",
    values="count", fill_value=0
)

# ─── 5. Select top-K by total errors ──────────────────────────────────────
top_gt   = conf_df.groupby("ground_truth")["count"].sum().nlargest(TOP_K).index
top_pred = conf_df.groupby("prediction")["count"].sum().nlargest(TOP_K).index
cm_sub   = cm.loc[top_gt, top_pred].astype(int)

cm_sub.to_csv(CSV_OUT, encoding="utf-8-sig")
print(f"✅ Confusion matrix subset saved to {CSV_OUT}")


# ─── 6. Plot the heatmap ─────────────────────────────────────────────────
# load Devanagari font so ticks and title render correctly
dev_font = fm.FontProperties(fname=FONT_PATH)

plt.figure(figsize=(12,10))
ax = sns.heatmap(
    cm_sub, annot=True, fmt="d",
    cmap="Reds", linewidths=0.5, linecolor="gray",
    cbar=True, square=True
)

ax.set_title("🔤 Top OCR Character Confusions", fontsize=16, fontproperties=dev_font)
ax.set_xlabel("Predicted Character", fontproperties=dev_font)
ax.set_ylabel("Ground Truth Character", fontproperties=dev_font)

# rotate tick labels and apply font
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=dev_font, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=dev_font, rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"✅ Heatmap saved to {OUTPUT_PNG}")