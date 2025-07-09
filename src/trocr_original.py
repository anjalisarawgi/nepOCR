import os
import argparse
import torch
import wandb
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from utils.data import load_dataset, OCRLazyDataset, collate_fn
from utils.callbacks import PrintPredictionsCallback
from utils.metrics import compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    set_seed(args.seed)

    if args.finetune_from_model:
        model_dir = os.path.join("models", f"finetuned_on_{args.dataset_name}")
    else:
        model_dir = os.path.join("models", f"scratch_on_{args.dataset_name}")

    os.makedirs(model_dir, exist_ok=True)

    # Init wandb with default run name
    wandb.init(project="nepOCR-logs", name="ocr-run")

    # Load dataset
    if args.dataset_name == "oldNepaliSynthetic":
        dataset = load_dataset("data/oldNepaliSynthetic10k/labels.json")
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "oldNepaliSynthetic30k":
        dataset = load_dataset("data/oldNepaliSynthetic30k/labels.json")
        split_dataset = dataset.train_test_split(train_size=30000, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "oldNepaliSynthetic_50k":
        dataset = load_dataset("data/oldNepaliSynthetic_50k/labels.json")
        split_dataset = dataset.train_test_split(train_size=50000, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        eval_dataset = eval_dataset.select(range(1000))
    elif args.dataset_name == "oldNepaliSynthetic_105k":
        dataset = load_dataset("data/oldNepaliSynthetic_105k/labels.json")
        split_dataset = dataset.train_test_split(train_size=80000, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        eval_dataset = eval_dataset.select(range(1000))
    elif args.dataset_name == "nagari":
        train_dataset = load_dataset("data/nagari/augmented3/train/labels.json")
        test_dataset = load_dataset("data/nagari/augmented3/test/labels.json")
        val_dataset = test_dataset
    elif args.dataset_name == "nagari_original":
        train_dataset = load_dataset("data/nagari/original/train/labels.json")
        eval_dataset = load_dataset("data/nagari/original/test/labels.json")
        test_dataset = eval_dataset
        val_dataset = eval_dataset
    elif args.dataset_name == "oldNepali_original":
        train_dataset = load_dataset("data/oldNepali/original/train/labels.json")
        eval_dataset = load_dataset("data/oldNepali/original/test/labels.json")
    elif args.dataset_name == "oldNepaliDataset_new3":
        train_dataset = load_dataset("data/oldNepaliDataset_new3/labels_train_processed.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed.json")
    elif args.dataset_name == "oldNepali":
        train_dataset = load_dataset("data/oldNepali/processed/labels_train_raw.json")
        test_dataset = load_dataset("data/oldNepali/processed/labels_test_raw.json")
        val_dataset = load_dataset("data/oldNepali/processed/labels_val_raw.json")
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # Load processor
    processor_path = args.finetune_from_model or args.trocr_model_name
    processor = TrOCRProcessor.from_pretrained(processor_path)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor

    print("\n[Tokenizer Debug]")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})\n")

    # Prepare datasets
    max_length = 256
    train_ds = OCRLazyDataset(train_dataset, tokenizer, feature_extractor, max_length=max_length)
    test_dataset = OCRLazyDataset(test_dataset, tokenizer, feature_extractor, max_length=max_length)
    val_dataset = OCRLazyDataset(val_dataset, tokenizer, feature_extractor, max_length=max_length)

    sample_batch = [test_dataset[i] for i in range(5)]
    print_callback = PrintPredictionsCallback(sample_batch, tokenizer, print_every=1000)

    # Load model
    model_path = args.finetune_from_model or args.trocr_model_name
    print(f"Loading model from: {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = max_length

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        warmup_steps=500,
        num_train_epochs=20,
        learning_rate=3e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        max_grad_norm=0.5,
        save_strategy="no",
        dataloader_num_workers=4,
        gradient_accumulation_steps=2,
        dataloader_pin_memory=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[print_callback]
    )

    trainer.train()
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    print("Final Evaluation:", trainer.evaluate(test_dataset))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--trocr_model_name", type=str,
                        choices=["microsoft/trocr-small-handwritten", "microsoft/trocr-base-handwritten", "microsoft/trocr-large-handwritten"],
                        default="microsoft/trocr-base-handwritten")
    parser.add_argument("--finetune_from_model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)