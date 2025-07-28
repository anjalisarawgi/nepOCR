import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import SequenceMatcher


CSV_PATH = "results/trocr_large_bert_byteBPE/predictions_with_logits.csv"

df = pd.read_csv(CSV_PATH, converters={
    "pred_tokens": ast.literal_eval,
    "pred_token_probs": ast.literal_eval,
    "gt_tokens": ast.literal_eval
})

### exact matching - if one if wrong all is wrong
# def analyze_token_level_confidence(row):
#     pred_tokens = row["pred_tokens"]
#     gt_tokens = row["gt_tokens"]
#     probs = row["pred_token_probs"]

#     min_len = min(len(pred_tokens), len(gt_tokens), len(probs))
#     return [
#         {"token": p, "gt": g, "prob": prob, "correct": p == g}
#         for p, g, prob in zip(pred_tokens[:min_len], gt_tokens[:min_len], probs[:min_len])
#     ]

# df["token_analysis"] = df.apply(analyze_token_level_confidence, axis=1)

### 
def align_and_score(pred_tokens, gt_tokens, probs):
    aligned = []
    matcher = SequenceMatcher(None, gt_tokens, pred_tokens)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if j < len(pred_tokens) and j < len(probs) and i < len(gt_tokens):
                    aligned.append({
                        "token": pred_tokens[j],
                        "gt": gt_tokens[i],
                        "prob": probs[j],
                        "correct": True
                    })
        else:
            for j in range(j1, j2):
                if j < len(pred_tokens) and j < len(probs):
                    aligned.append({
                        "token": pred_tokens[j],
                        "gt": None,
                        "prob": probs[j],
                        "correct": False
                    })
    return aligned

df["token_analysis"] = df.apply(lambda row: align_and_score(row["pred_tokens"], row["gt_tokens"], row["pred_token_probs"]), axis=1)

token_df = pd.DataFrame([t for sublist in df["token_analysis"] for t in sublist])
total_tokens = len(token_df)
correct_tokens = token_df["correct"].sum()
incorrect_tokens = total_tokens - correct_tokens
print("Total tokens:", total_tokens)
print("Correct tokens:", correct_tokens)
print("Incorrect tokens:", incorrect_tokens)

# Bins
bins = np.arange(0.0, 1.01, 0.1)
labels = [f"{round(bins[i],1)}–{round(bins[i+1],1)}" for i in range(len(bins)-1)]
token_df["prob_bin"] = pd.cut(token_df["prob"], bins=bins, labels=labels, include_lowest=True)

token_df.to_csv("results/trocr_large_bert_byteBPE/token_df.csv", index=False)
bin_counts = token_df.groupby(["prob_bin", "correct"]).size().unstack(fill_value=0)
bin_counts.to_csv("results/trocr_large_bert_byteBPE/token_prob_analysis.csv")
print(bin_counts)

