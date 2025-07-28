import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
from torchmetrics.functional.text import char_error_rate
import unicodedata
from torch.nn.functional import softmax


MODEL_DIR = "models/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_fullset_aug8"
TEST_LABELS_PATH = "data/oldNepali_fullset/labels_normalized_final/labels_test.json"
OUTPUT_CSV = "results/trocr_large_bert_byteBPE/predictions_with_logits.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

# model and processor
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
processor = TrOCRProcessor(
    image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
    tokenizer=tokenizer
)
model.config.update({
    "decoder_start_token_id": tokenizer.cls_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_length": 256,
    "num_beams": 5,
    "early_stopping": True,
    "no_repeat_ngram_size": 0
})

    
with open(TEST_LABELS_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    # test_data = test_data[:10]

# predicts both text and saves also the individual token probabilities
def predict_text_with_token_probs(image_path):
    img = Image.open(image_path)
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=5, 
            no_repeat_ngram_size=0,
            max_length=256,
        )

    pred_ids = output.sequences[0]
    pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
    token_logits = output.scores  # logits before softmax
    
    # calculate probabilities for each token using softmax
    token_probs = []
    for i, logits in enumerate(token_logits):
        if i + 1 < len(pred_ids): # because first token is the start token
            probs = softmax(logits[0], dim=-1)
            token_id = pred_ids[i + 1].item()
            prob = probs[token_id].item()
            token_probs.append(prob)

    return pred_text, pred_ids.tolist(), token_probs


results = []
for sample in tqdm(test_data):
    image_path = sample["image_path"]
    gt = sample["text"]
    pred, pred_token_ids, pred_token_probs = predict_text_with_token_probs(image_path)
    gt = clean_text(gt)
    pred = clean_text(pred)
    cer  = char_error_rate([pred],[gt]).item()

    # token ids 
    gt_token_ids = tokenizer.encode(gt, add_special_tokens=False)
    pred_token_ids = tokenizer.encode(pred, add_special_tokens=False)

    # tokens (also converting for readability)
    gt_tokens = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokenizer.convert_ids_to_tokens(gt_token_ids)] # converting to readable tokens
    pred_tokens = [tokenizer.convert_tokens_to_string([t]).strip() for t in tokenizer.convert_ids_to_tokens(pred_token_ids)]

    results.append({
        "image_path": image_path,
        "ground_truth": gt,
        "prediction": pred,
        "cer": cer, 
        "gt_tokens": gt_tokens,
        "pred_tokens": pred_tokens, 
        "gt_token_ids": gt_token_ids,
        "pred_token_ids": pred_token_ids,
        "pred_token_probs": pred_token_probs
    })


df = pd.DataFrame(results)

# weighted cer
df["gt_length"] = df["ground_truth"].apply(lambda txt: len(clean_text(txt))) # length of ground truth text
df["weighted_errors"] = df["cer"] * df["gt_length"]
total_chars = df["gt_length"].sum()
total_errs = df["weighted_errors"].sum()
weighted_cer = total_errs/total_chars

df.to_csv(OUTPUT_CSV, index=False)
print("Results saved to", OUTPUT_CSV)
print("weighted CER:", weighted_cer)