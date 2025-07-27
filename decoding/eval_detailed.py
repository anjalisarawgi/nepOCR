# import os
# import json
# import re
# import torch
# import pandas as pd
# from tqdm import tqdm
# from PIL import Image
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, AutoImageProcessor
# from torchmetrics.text import CharErrorRate

# MODEL_DIR = "models/trocr-base-handwritten-BERT-oldNepaliSynthetic_105k_vnoisy-byteBPE-500_finetuned_on_nagari_finetuned_on_oldNepaliDataset_new_42"
# TEST_LABELS_PATH = "data/oldNepaliDataset_new_42/labels_test.json"
# ROOT_OUTPUT_DIR = "decoding/results/beam_search_detailed"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLEANUP = re.compile(r'[\u00AD\u200B\u200C\u200D]')

# def clean_text(text):
#     return CLEANUP.sub('', text)


# def run_beam_search_experiment(model, processor, tokenizer, test_labels):
#     os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)
#     results_path = os.path.join(ROOT_OUTPUT_DIR, 'results.csv')
#     json_path = os.path.join(ROOT_OUTPUT_DIR, 'results.json')

#     results = []
#     cer_metric = CharErrorRate()

#     for sample in tqdm(test_labels):
#         image_path = sample['image_path']
#         ground_truth = clean_text(sample['text'])

#         image = Image.open(image_path).convert("RGB")
#         pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

#         with torch.no_grad():
#             output = model.generate(
#                 pixel_values,
#                 max_length=256,
#                 num_beams=10,
#                 num_return_sequences=1,
#                 output_scores=True, # for entropy calculation
#                 return_dict_in_generate=True
#             )

#         output_ids = output.sequences
#         pred_ids = output_ids[0].tolist()
#         pred_text = clean_text(tokenizer.decode(pred_ids, skip_special_tokens=True))

#         # Compute average entropy over tokens
#         token_scores = output.scores  # list of tensors, each (vocab_size,)
#         entropies = []

#         for i, logits in enumerate(token_scores):
#             probs = torch.softmax(logits[0], dim=-1)  # (vocab_size,)
#             log_probs = torch.log_softmax(logits[0], dim=-1)
#             entropy = -(probs * log_probs).sum().item()
#             entropies.append(entropy)

#         avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        
#         # Compute token-level info
#         token_infos = []
#         token_scores = output.scores  # list of logits per timestep
#         pred_ids = output.sequences[0].tolist()

#         for i, token_id in enumerate(pred_ids[1:]):  # skip decoder_start_token
#             if i >= len(token_scores):
#                 break

#             logits = token_scores[i][0]  # shape: (vocab_size,)
#             probs = torch.softmax(logits, dim=-1)
#             log_probs = torch.log_softmax(logits, dim=-1)
#             entropy = -(probs * log_probs).sum().item()

#             topk_probs, topk_ids = torch.topk(probs, k=5)
#             topk_logits = logits[topk_ids]
#             topk_log_probs = log_probs[topk_ids]
#             topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids.tolist())
#             topk_strs = [tokenizer.convert_tokens_to_string([t]).strip() for t in topk_tokens]

#             topk_info = []
#             for s, p, l, lp in zip(topk_strs, topk_probs, topk_logits, topk_log_probs):
#                 topk_info.append({
#                     "token": s,
#                     "prob": p.item(),
#                     "logit": l.item(),
#                     "log_prob": lp.item()
#                 })

#             pred_token = tokenizer.convert_tokens_to_string(
#                 [tokenizer.convert_ids_to_tokens([token_id])[0]]
#             ).strip()

#             token_infos.append({
#                 "pred_token": pred_token,
#                 "entropy": entropy,
#                 "topk": topk_info
#             })

#         cer_score = cer_metric(pred_text, ground_truth).item()
#         results.append({
#             'image_path': image_path,
#             'ground_truth': ground_truth,
#             'pred_text': pred_text,
#             'cer': cer_score,
#             'avg_entropy': avg_entropy,
#             'token_probs': token_infos
#         })

#     # CSV
#     df = pd.DataFrame(results)
#     df.to_csv(results_path, index=False, encoding='utf-8')

#     #  JSON
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     with open(os.path.join(ROOT_OUTPUT_DIR, 'results_with_tokens.json'), 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     corpus_cer = cer_metric(
#         [r["pred_text"] for r in results],
#         [r["ground_truth"] for r in results]
#     ).item()
#     print(f"\n Beam Search | Corpus CER: {corpus_cer:.4f}")

# def main():
#     print("Loading model...")
#     model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     processor = TrOCRProcessor(
#         image_processor=AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
#         tokenizer=tokenizer
#     )
#     model.config.decoder_start_token_id = tokenizer.cls_token_id
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id
#     model.config.max_length = 256

#     with open(TEST_LABELS_PATH, 'r', encoding='utf-8') as f:
#         test_labels = json.load(f)

#     run_beam_search_experiment(model, processor, tokenizer, test_labels)

# if __name__ == "__main__":
#     main()