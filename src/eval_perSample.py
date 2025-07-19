import re
import unicodedata
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
from torchmetrics.functional.text import char_error_rate

# --- Config ---
MODEL_DIR = "models/trained/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_aug16"
IMAGE_PATH = "sample_1.png"
GROUND_TRUTH = "your expected ground truth text here"
MAX_LENGTH = 256
NUM_BEAMS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cleanup regex ---
CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

# --- Load model and processor ---
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

# --- Predict ---
def predict_text(image_path):
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- Run on one image ---
prediction = predict_text(IMAGE_PATH)
gt_clean = clean_text(GROUND_TRUTH)
pred_clean = clean_text(prediction)

# --- Compute CER ---
raw_cer = char_error_rate([prediction], [GROUND_TRUTH]).item()
cleaned_cer = char_error_rate([pred_clean], [gt_clean]).item()

# --- Output ---
print("\n🔍 Evaluation for single image")
print(f"📷 Image:         {IMAGE_PATH}")
print(f"🧾 Ground Truth:  {GROUND_TRUTH}")
print(f"🤖 Prediction:    {prediction}")
print(f"❌ Raw CER:       {raw_cer:.4f}")
print(f"🧼 Cleaned CER:   {cleaned_cer:.4f}")