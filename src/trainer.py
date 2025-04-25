import os
import argparse
import torch
import wandb
from transformers import (
    VisionEncoderDecoderModel,
    BertLMHeadModel,
    BertConfig,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)

from utils.data import load_dataset, preprocess_dataset, OCRTorchDataset, collate_fn
from utils.tokenizer import train_tokenizer
from utils.callbacks import PrintPredictionsCallback
from utils.metrics import compute_metrics
# from utils.seed import set_all_seeds

def main(args):
    set_seed(args.seed)
    # set_all_seeds(args.seed)
    wandb.init(project=args.wandb_project, name=args.wandb_run)

    # dataset
    dataset = load_dataset(args.json_path)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    train_dataset = train_dataset.select(range(200))
    eval_dataset = eval_dataset.select(range(50))

    # tokenizer
    tokenizer = train_tokenizer(
        corpus_path=args.corpus_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
        output_dir=args.tokenizer_dir
    )

    # processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    feature_extractor = processor.feature_extractor


    train_imgs, train_lbls = preprocess_dataset(train_dataset, tokenizer, feature_extractor, args.max_length)
    eval_imgs, eval_lbls = preprocess_dataset(eval_dataset, tokenizer, feature_extractor, args.max_length)

    train_ds = OCRTorchDataset(train_imgs, train_lbls)
    eval_ds = OCRTorchDataset(eval_imgs, eval_lbls)

    # callback initialized for debugging and inference during traning
    sample_batch = [eval_ds[i] for i in range(5)]
    print_callback = PrintPredictionsCallback(sample_batch, tokenizer, print_every=100)


    # model
    encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").encoder
    decoder_config = BertConfig(
        is_decoder=True,
        add_cross_attention=True,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoder = BertLMHeadModel(decoder_config)
    
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # configs for the model
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.no_repeat_ngram_size = 5
    model.config.num_beams = 5
    model.config.early_stopping = True

    # training 
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=100,
        warmup_steps=500,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        max_grad_norm=0.5,
        save_strategy="no",
        gradient_accumulation_steps=2 if torch.cuda.is_available() else 1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,  
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[print_callback]
    )

    trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print("Final Evaluation:", trainer.evaluate(eval_ds))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="oldNepali-ocr-logs")
    parser.add_argument("--wandb_run", type=str, default="nepOCR-logs")
    parser.add_argument("--dataset_name", type=str, default="oldNepaliSynthetic10k")
    parser.add_argument("--json_path", type=str, default="data/oldNepaliSynthetic/10k/labels_processed.json")
    parser.add_argument("--corpus_path", type=str, default="corpus/oldNepaliSynthetic_nagari_oldNepali.txt")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/charBPE/")

    # model setup args
    parser.add_argument("--encoder", type=str, choices=["trocr"], default="trocr")
    parser.add_argument("--decoder", type=str, choices=["bert"], default="bert")
    parser.add_argument("--tokenizer_type", type=str, choices=["char", "sbpe"], default="char")
    parser.add_argument("--vocab_size", type=int, default=1000)

    # training args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.model_name = f"{args.encoder}-{args.decoder.upper()}-{args.dataset_name}-{args.tokenizer_type}-{args.vocab_size}"
    args.model_dir = os.path.join("models", args.model_name)
    args.wandb_run = args.model_name

    main(args)

