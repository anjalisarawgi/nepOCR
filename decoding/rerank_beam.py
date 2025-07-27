# import os
# import json
# import re
# import torch
# import pandas as pd
# from tqdm import tqdm
# from PIL import Image
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
# from torchmetrics.text import CharErrorRate
# from collections import Counter

# MODEL_DIR = "models/trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepaliDataset_new_42"
# TEST_LABELS_PATH = "data/oldNepaliDataset_new_42/labels_test.json"
# OUTPUT_DIR = "decoding/results/beam_search_reranked"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# cer_metric = CharErrorRate()
# print(f"Loading model from {MODEL_DIR}...")
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# processor = TrOCRProcessor(
#     image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
#     tokenizer=tokenizer
# )
# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 256

# # test data
# with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
#     test_labels = json.load(f)

# def predict_beam_search(pixel_values):
#     with torch.no_grad():
#         output = model.generate(
#             pixel_values=pixel_values,
#             num_beams=20,
#             num_return_sequences=20,
#             output_scores=True,
#             return_dict_in_generate=True,
#             max_length=256
#         )
#         return output

# results = []
# beam_selection_counter = Counter()

# # rerank beam search
# for sample in tqdm(test_labels, desc="Evaluating with beam search + reranking"):
#     image = Image.open(sample['image_path']).convert("RGB")
#     gt = CLEANUP.sub('', sample['text'])
#     pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

#     output = predict_beam_search(pixel_values)
#     output_ids = output.sequences
#     decoded_texts = [CLEANUP.sub('', t) for t in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]

#     cer_scores = [cer_metric(pred, gt).item() for pred in decoded_texts] # compute cer for each beam 
#     best_idx = cer_scores.index(min(cer_scores)) # best beam 

#     results.append({
#         "image_path": sample['image_path'],
#         "ground_truth": gt,
#         "reranked_text": decoded_texts[best_idx],
#         "cer": cer_scores[best_idx],
#         "selected_beam_index": best_idx,
#         "beam_outputs": decoded_texts
#     })
#     beam_selection_counter[best_idx] += 1


# results_path = os.path.join(OUTPUT_DIR, 'results.csv')
# json_path = os.path.join(OUTPUT_DIR, 'results.json')
# beam_count_path = os.path.join(OUTPUT_DIR, 'beam_selection_counts.json')

# pd.DataFrame(results).to_csv(results_path, index=False, encoding='utf-8')
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
# with open(beam_count_path, 'w', encoding='utf-8') as f:
#     json.dump(dict(beam_selection_counter), f, indent=2)

# # ——— Report ———
# perfect = sum(1 for r in results if r['cer'] == 0.0)
# print(f"accuracy: {perfect}/{len(results)} ({(perfect/len(results))*100:.2f}%)")
# corpus_cer = cer_metric([r['reranked_text'] for r in results], [r['ground_truth'] for r in results]).item()
# print(f"Corpus CER: {corpus_cer:.4f}")