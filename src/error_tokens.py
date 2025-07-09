import pandas as pd
import difflib
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# === Load your CSV ===
df = pd.read_csv("results/evaluation_results_cleaned_bytebpe500_test_new8.csv")

# === Count total GT characters ===
gt_total_counter = Counter()
for text in df['ground_truth']:
    gt_total_counter.update(str(text))

# === Count mispredicted GT characters (replacements + deletions) and insertions ===
gt_error_counter = Counter()
pred_insert_counter = Counter()

def analyze_errors(gt, pred):
    sm = difflib.SequenceMatcher(None, gt, pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('replace', 'delete'):
            gt_error_counter.update(gt[i1:i2])
        if tag in ('replace', 'insert'):
            pred_insert_counter.update(pred[j1:j2])

for _, row in df.iterrows():
    gt = str(row['ground_truth'])
    pred = str(row['prediction'])
    analyze_errors(gt, pred)

# === Merge all characters seen anywhere ===
all_chars = set(gt_total_counter) | set(gt_error_counter) | set(pred_insert_counter)

# === Build the final report ===
records = []
for char in all_chars:
    total = gt_total_counter.get(char, 0)
    errors = gt_error_counter.get(char, 0)
    insertions = pred_insert_counter.get(char, 0)
    error_rate = errors / total if total > 0 else 0
    was_inserted_only = total == 0  # hallucination only
    records.append({
        "char": char,
        "total_count": total,
        "error_count": errors,
        "insertion_count": insertions,
        "error_rate": error_rate,
        "was_inserted_only": was_inserted_only
    })

df_report = pd.DataFrame(records)
df_report = df_report.sort_values(by=["error_count", "insertion_count", "error_rate"], ascending=[False, False, False])
df_report.to_csv("results/full_char_error_report.csv", index=False)

print("🔍 Top 20 error-prone or hallucinated characters:")
print(df_report.head(20))

# === Plotting section ===

# Step 1: Load Devanagari-compatible font
font_path = "fonts/NotoSansDevanagari-Regular.ttf"  # <-- update path if needed
devanagari_font = fm.FontProperties(fname=font_path)

# Step 2: Plot top 20 error characters
top = df_report.sort_values("error_count", ascending=False).head(20)

plt.figure(figsize=(12, 6))
bars = plt.bar(top["char"], top["error_count"], color="red")

# Step 3: Annotate bars with count
for bar, count in zip(bars, top["error_count"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(count),
             ha='center', va='bottom', fontproperties=devanagari_font)

# Step 4: Labels and title
plt.title("🔤 Top 20 OCR Error Characters", fontproperties=devanagari_font, fontsize=16)
plt.xlabel("अक्षर (Character)", fontproperties=devanagari_font, fontsize=12)
plt.ylabel("Error Count", fontproperties=devanagari_font, fontsize=12)
plt.xticks(fontproperties=devanagari_font, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Step 5: Save output
plt.savefig("results/top_20_error_characters.png", dpi=300, bbox_inches='tight')
print("✅ Saved chart to results/top_20_error_characters.png")
plt.show()