import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
from torchmetrics.text import CharErrorRate
from argparse import ArgumentParser

MODEL_DIR = "models/trocr-large-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepali_fullset_aug8"
TEST_LABELS_PATH = "data/oldNepali_fullset/labels_normalized/labels_test.json"
ROOT_OUTPUT_DIR = "decoding/results/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')

def clean_text(text):
    return CLEANUP.sub('', text)

def predict(model, pixel_values, strategy, config):
    with torch.no_grad():
        if strategy == "beam_search":
            return model.generate(
                pixel_values, 
                max_length=256, 
                num_beams=10,
                # num_return_sequences=10
                )

        elif strategy == "contrastive":
            k, alpha = config
            return model.generate(
                pixel_values, 
                max_length=256, 
                top_k=k, 
                penalty_alpha=alpha
                )

        elif strategy == "temp_sampling":
            tau = config
            return model.generate(
                pixel_values, 
                max_length=256, 
                do_sample=True, 
                temperature=tau
                )

        elif strategy == "top_k":
            return model.generate(
                pixel_values, 
                max_length=256, 
                do_sample=True, 
                top_k=config
                )

        elif strategy == "top_p":
            return model.generate(
                pixel_values, 
                max_length=256, 
                do_sample=True, 
                top_p=config
                )

def run_experiment(model, processor, tokenizer, strategy, config_name, config_value, test_labels):
    output_dir = os.path.join(ROOT_OUTPUT_DIR, strategy, config_name)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.csv')
    json_path = os.path.join(output_dir, 'results.json')

    results = []
    cer_metric = CharErrorRate()

    for sample in tqdm(test_labels, desc=f"{strategy}:{config_name}"):
        image_path = sample['image_path']
        ground_truth = clean_text(sample['text'])

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

        output_ids = predict(model, pixel_values, strategy, config_value)
        pred_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        pred_text = clean_text(pred_text)

        cer_score = cer_metric(pred_text, ground_truth).item()
        results.append({
            'image_path': image_path,
            'ground_truth': ground_truth,
            'pred_text': pred_text,
            'cer': cer_score
        })


    df = pd.DataFrame(results)

    # Save per-line results
    df.to_csv(results_path, index=False, encoding='utf-8')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Corpus-level CER
    corpus_cer = cer_metric(
        [r["pred_text"] for r in results],
        [r["ground_truth"] for r in results]
    ).item()

    # Mean of per-line CERs
    mean_line_cer = df["cer"].mean()

    print(f"{strategy} | {config_name} | Corpus CER: {corpus_cer:.4f} | Mean Line CER: {mean_line_cer:.4f}")

def main():
    # model and processor
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    processor = TrOCRProcessor(
        image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
        tokenizer=tokenizer
    )
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 256

    with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
        test_labels = json.load(f)

    # hyperparameters
    grid = {
        "beam_search": {
            f"beam{b}": b for b in [1, 5, 10, 20]
        },
        "contrastive": {
            f"k{k}_alpha{a}": (k, a) for k in [5, 10] for a in [0.2, 0.6, 0.8]
        },
        "temp_sampling": {
            f"tau{t}": t for t in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        },
        "top_k": {
            f"topk{k}": k for k in [3, 5, 10, 20, 50]
        },
        "top_p": {
            f"topp{int(p*100)}": p for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        }, 
        # "beam_search_with_diversity": {
        #     f"beam{b}_diversity{d}": (b, d) for b in [1, 5, 10] for d in [0.2, 0.5, 0.8, 1.0]
        # }
    }

    for strategy, configs in grid.items():
        for config_name, config_val in configs.items():
            run_experiment(model, processor, tokenizer, strategy, config_name, config_val, test_labels)

if __name__ == "__main__":
    main()