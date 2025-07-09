import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

JSON_PATH = "decoding/results/beam_search_detailed/results_with_tokens.json"
PROB_THRESHOLD = 0.5

def align_pred_with_truth(pred, truth):
    aligned = []
    min_len = min(len(pred), len(truth))
    for i in range(min_len):
        aligned.append(pred[i] == truth[i])
    aligned.extend([False] * (len(pred) - min_len))
    return aligned

with open(JSON_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

rows = []

for r in tqdm(results):
    pred = r["pred_text"]
    truth = r["ground_truth"]
    tokens = r["token_probs"] 

    correctness = align_pred_with_truth(pred, truth)

    for i, token in enumerate(tokens):
        is_wrong = True if i >= len(correctness) else not correctness[i]
        entropy = token["entropy"]
        top_prob = token["topk"][0]["prob"]  

        rows.append({
            "image_path": r["image_path"],
            "token_index": i,
            "pred_token": token["pred_token"],
            "entropy": entropy,
            "top_prob": top_prob,
            "is_wrong": is_wrong
        })

df = pd.DataFrame(rows)
low_prob_df = df[df["top_prob"] < PROB_THRESHOLD]

wrong_low_prob = low_prob_df["is_wrong"].sum()
total_low_prob = len(low_prob_df)

print(f"Low-probability tokens (< {PROB_THRESHOLD}): {total_low_prob}")
print(f"Wrong predictions among them: {wrong_low_prob}")
if total_low_prob > 0:
    error_pct = 100 * wrong_low_prob / total_low_prob
    print(f"percentage of wrong among low-prob tokens!!!: {error_pct:.2f}%")
else:
    print("No tokens below probability threshold.")


total_wrong_preds = df["is_wrong"].sum()
wrong_with_low_prob = low_prob_df["is_wrong"].sum()
print(f"Total wrong predictions: {total_wrong_preds}")
print(f"Wrong predictions with low prob (< {PROB_THRESHOLD}): {wrong_with_low_prob}")
print(f"Fraction of wrong preds that are low-prob: {100 * wrong_with_low_prob / total_wrong_preds:.2f}%")