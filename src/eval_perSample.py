# import re
# import unicodedata
# import torch
# from PIL import Image
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
# from torchmetrics.functional.text import char_error_rate


# MODEL_DIR = "models/trained/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_aug16"
# IMAGE_PATH = "sample_1.png"
# OUTPUT_FILE = "ocr_predictions.txt"
# MAX_LENGTH = 256
# NUM_BEAMS = 5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
# # def clean_text(text: str) -> str:
# #     text = unicodedata.normalize("NFC", text)
# #     text = CLEANUP.sub("", text)
# #     return re.sub(r"\s+", "", text)

# model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# processor = TrOCRProcessor(
#     image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-large-handwritten"),
#     tokenizer=tokenizer
# )
# model.config.update({
#     "decoder_start_token_id": tokenizer.cls_token_id,
#     "pad_token_id":           tokenizer.pad_token_id,
#     "eos_token_id":           tokenizer.eos_token_id,
#     "max_length":             MAX_LENGTH,
#     "num_beams":              NUM_BEAMS,
#     "early_stopping":         True,
#     "no_repeat_ngram_size":   0
# })

# def predict_text(image_path):
#     img = Image.open(image_path).convert("RGB")
#     pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
#     with torch.no_grad():
#         generated_ids = model.generate(pixel_values)
#     return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# prediction = predict_text(IMAGE_PATH)
# # pred_clean = clean_text(prediction)

# print(f" Prediction:    {prediction}")


import os
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor

MODEL_DIR = "models/trained/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_181_aug8"
IMAGE_FOLDER = "test_samples/DNA_0009_0158"
OUTPUT_FILE = "test_samples/DNA_0009_0158/predictions_DNA_0009_0158.txt"
MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model and the processor
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
processor = TrOCRProcessor(
    image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-large-handwritten"),
    tokenizer=tokenizer
)

model.config.update({
    "decoder_start_token_id": tokenizer.cls_token_id,
    "pad_token_id":           tokenizer.pad_token_id,
    "eos_token_id":           tokenizer.eos_token_id,
    "max_length":             MAX_LENGTH,
    "num_beams":              NUM_BEAMS,
    "early_stopping":         True,
    "no_repeat_ngram_size":   0
})


def predict_text(image_path):
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

results = []
for fname in sorted(os.listdir(IMAGE_FOLDER)):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(IMAGE_FOLDER, fname)
        try:
            prediction = predict_text(img_path)
            results.append(f"{fname} | {prediction}")
            print(f"{fname} → {prediction}")
        except Exception as e:
            print(f" Error on {fname}: {e}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"All predictions saved to {OUTPUT_FILE}")