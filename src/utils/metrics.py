from torchmetrics import CharErrorRate

def compute_metrics(predictions, tokenizer):
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_od 
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)
    return {"cer": CharErrorRate()(pred_str, label_str)}