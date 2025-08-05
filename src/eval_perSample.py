import re
import unicodedata
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
from torchmetrics.functional.text import char_error_rate


model_dir = "models/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_fullset_aug8"
# IMAGE_PATH = "data/oldNepali_fullset/oldNepali/images/DNA_0014_0296 zuschn/eSc_line_b278b45e.png"
image = "data/oldNepali_fullset/oldNepali/images/DNA_0014_0296 zuschn/eSc_line_996ee0ca.png"
# output_file = "ocr_predictions.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CLEANUP.sub("", text)
    return re.sub(r"\s+", "", text)

model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
processor = TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-large-handwritten"),tokenizer=tokenizer)
model.config.update({
    "decoder_start_token_id": tokenizer.cls_token_id,
    "pad_token_id":           tokenizer.pad_token_id,
    "eos_token_id":           tokenizer.eos_token_id,
    "max_length":             256,
    "num_beams":              5,
    "early_stopping":         True,
    "no_repeat_ngram_size":   0
})

def predict_text(image_path):
    img = Image.open(image_path).convert("L")  
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

prediction = predict_text(image)
# pred_clean = clean_text(prediction)

print("Prediction:" prediction)
