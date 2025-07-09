from transformers import AutoTokenizer
import json
import pandas as pd
from difflib import SequenceMatcher

tokenizer = AutoTokenizer.from_pretrained(
    "models/trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_"
    "finetuned_on_nagari_finetuned_on_oldNepaliDataset_new_42"
)

with open("decoding/results/beam_search_detailed/results_with_tokens.json",
          "r", encoding="utf-8") as f:
    data = json.load(f)

errors_results = []
corrects_results = []

for item in data:
    gt          = item["ground_truth"]
    pred        = item["pred_text"]
    cer         = item.get("cer", None)
    token_probs = item.get("token_probs", []) 
    gt_ids        = tokenizer(gt, add_special_tokens=False).input_ids
    pred_ids      = tokenizer(pred, add_special_tokens=False).input_ids
    gt_subwords   = [tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                     for tid in gt_ids]
    pred_subwords = [tokenizer.decode([pid], clean_up_tokenization_spaces=False)
                     for pid in pred_ids]

    sm = SequenceMatcher(None, gt_subwords, pred_subwords)
    edits = 0

    err_toks, err_ents, err_probs = [], [], []
    cor_toks, cor_ents, cor_probs = [], [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # all matching tokens
            for k in range(j1, j2):
                tp = token_probs[k]
                cor_toks.append(pred_subwords[k])
                cor_ents.append(tp["entropy"])
                cor_probs.append(tp["topk"][0]["prob"])
        elif tag == "replace":
            n_eq = min(i2 - i1, j2 - j1)
            edits += n_eq
            for k in range(n_eq):
                idx = j1 + k
                tp  = token_probs[idx]
                err_toks.append(pred_subwords[idx])
                err_ents.append(tp["entropy"])
                err_probs.append(tp["topk"][0]["prob"])
            for k in range(n_eq, j2 - j1):
                idx = j1 + k
                tp  = token_probs[idx]
                edits += 1
                err_toks.append(pred_subwords[idx])
                err_ents.append(tp["entropy"])
                err_probs.append(tp["topk"][0]["prob"])
        elif tag == "delete":
            edits += (i2 - i1)
        elif tag == "insert":
            for k in range(j1, j2):
                tp = token_probs[k]
                edits += 1
                err_toks.append(pred_subwords[k])
                err_ents.append(tp["entropy"])
                err_probs.append(tp["topk"][0]["prob"])

    ter = edits / len(gt_subwords) if gt_subwords else 0

    base = {
        "image_path"       : item.get("image_path", ""),
        "num_gt_tokens"    : len(gt_subwords),
        "num_pred_tokens"  : len(pred_subwords),
        "token_edits"      : edits,
        "token_error_rate" : ter,
        "cer"              : cer,
    }

    errors_results.append({
        **base,
        "error_tokens"     : "|".join(err_toks),
        "error_entropies"  : "|".join(str(e) for e in err_ents),
        "error_probs"      : "|".join(str(p) for p in err_probs),
    })

    corrects_results.append({
        **base,
        "correct_tokens"   : "|".join(cor_toks),
        "correct_entropies": "|".join(str(e) for e in cor_ents),
        "correct_probs"    : "|".join(str(p) for p in cor_probs),
    })

# 3) Build DataFrames
errors_df   = pd.DataFrame(errors_results)
corrects_df = pd.DataFrame(corrects_results)

# un-truncate for console if desired
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

print("=== Error Tokens ===")
print(errors_df.head())
print("\n=== Correct Tokens ===")
print(corrects_df.head())

# 4) Save to CSV
errors_df.to_csv("token_errors.csv", index=False, encoding="utf-8")
corrects_df.to_csv("token_corrects.csv", index=False, encoding="utf-8")

# 5) Save to JSON
with open("token_errors.json", "w", encoding="utf-8") as f:
    json.dump(errors_results, f, ensure_ascii=False, indent=2)
with open("token_corrects.json", "w", encoding="utf-8") as f:
    json.dump(corrects_results, f, ensure_ascii=False, indent=2)

print("→ Saved token_errors.csv/.json and token_corrects.csv/.json")