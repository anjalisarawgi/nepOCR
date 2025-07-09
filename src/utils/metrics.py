import re
import torch
from torchmetrics.text import CharErrorRate
from torchmetrics.functional.text import edit_distance
from rapidfuzz.distance import Levenshtein


cer_metric = CharErrorRate()
CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')

def clean_text(s: str) -> str:
    return CLEANUP.sub("", s)

def compute_metrics(predictions, tokenizer):
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [clean_text(p) for p in pred_str]
    label_str = [clean_text(l) for l in label_str]

    # CER
    cer = cer_metric(pred_str, label_str)

    # perfect predictions accuracy
    exact_matches = sum(p == l for p, l in zip(pred_str, label_str))
    acc = exact_matches / len(label_str) if label_str else 0.0

    return {"cer": cer.item(), "acc": acc}
