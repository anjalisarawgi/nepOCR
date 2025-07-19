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
    set_seed, 
    PreTrainedTokenizerFast, 
    GPT2Config,
    GPT2LMHeadModel,
    VisionEncoderDecoderConfig,
    SwinConfig,
    AutoFeatureExtractor, 
    SwinModel
)
from utils.data import load_dataset, preprocess_dataset, OCRTorchDataset, collate_fn, OCRLazyDataset
from utils.tokenizer import train_tokenizer
from utils.callbacks import PrintPredictionsCallback
from utils.metrics import compute_metrics
# from utils.seed import set_all_seeds
import time
import csv
import GPUtil
from datetime import datetime
from transformers import EarlyStoppingCallback


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    set_seed(args.seed)
    wandb.init(project="nepOCR-logs", name = args.model_name)

    # dataset
    if args.dataset_name == "oldNepaliSynthetic_105k_vnoisy":
        dataset = load_dataset("data/oldNepaliSynthetic_105k_vnoisy/labels_processed.json")
        split_dataset = dataset.train_test_split(train_size=100000, seed=args.seed)
        train_dataset = split_dataset["train"]
        full_eval_dataset = split_dataset["test"]
        eval_dataset = full_eval_dataset.select(range(0, 2500))
        val_dataset = eval_dataset
        test_dataset = full_eval_dataset.select(range(2500, len(full_eval_dataset)))
    elif args.dataset_name == "nagari_original":
        train_dataset = load_dataset("data/nagari/original/train/labels.json")
        eval_dataset = load_dataset("data/nagari/original/test/labels.json")
    elif args.dataset_name == "nagari":
        train_dataset = load_dataset("data/nagari/augmented4/train/labels_train.json")
        val_dataset = load_dataset("data/nagari/original/train/labels_val.json")
        test_dataset = load_dataset("data/nagari/augmented4/test/labels_test.json")
    elif args.dataset_name == "oldNepali_original":
        train_dataset = load_dataset("data/oldNepali/processed/raw_labels/labels_train_raw.json")
        eval_dataset = load_dataset("data/oldNepali/processed/raw_labels/labels_test_raw.json")
        test_dataset = eval_dataset
        val_dataset = load_dataset("data/oldNepali/processed/raw_labels/labels_val_raw.json")
    elif args.dataset_name == "oldNepali_aug16":
        train_dataset = load_dataset("data/oldNepali_aug16/labels_train.json")
        eval_dataset = load_dataset("data/oldNepali/processed/labels_test.json")
        test_dataset = eval_dataset
        val_dataset = load_dataset("data/oldNepali/processed/labels_val.json")

    
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")


    # slice for debugging
    # train_dataset = train_dataset.select(range(200))
    # eval_dataset = eval_dataset.select(range(50))

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval  dataset size: {len(val_dataset)} examples\n")
    print(f"Test  dataset size: {len(test_dataset)} examples")

    # tokenizer
    if args.finetune_from_model: # if finetuning from a pretrained model, load the tokenizer from the model (esp for nagari and oldNepali)
        model_path = args.finetune_from_model
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    else:
        tokenizer_dir = os.path.join("tokenizer", args.tokenizer_type + "_" + str(args.vocab_size))
        print(f"Loading tokenizer from: {tokenizer_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)


    print("\n Debugging and checking the tokenizer:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})\n")

    # processor
    model_path = "microsoft/" + args.encoder
    print("Loading processor from: ", model_path)

    if args.encoder == "swin" or args.encoder == "swin_from_scratch":
        print("using swin feature extractor")
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    else:
        print("using trOCR feature extractor")
        processor = TrOCRProcessor.from_pretrained(model_path)
        feature_extractor = processor.feature_extractor

    train_ds = OCRLazyDataset(train_dataset, tokenizer, feature_extractor, max_length=256)
    test_ds = OCRLazyDataset(test_dataset, tokenizer, feature_extractor, max_length=256)
    val_ds = OCRLazyDataset(val_dataset, tokenizer, feature_extractor, max_length=256)

    # callback for printing predictions for debugging
    sample_batch = [test_ds[i] for i in range(5)]
    print_callback = PrintPredictionsCallback(sample_batch, tokenizer, print_every=10000)
    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # or any patience value you prefer
        early_stopping_threshold=0.0  # how much improvement is considered significant
    )



    # model    
    if args.finetune_from_model:
        print(f"Finetuning from: {args.finetune_from_model}")
        model = VisionEncoderDecoderModel.from_pretrained(args.finetune_from_model)
    elif args.encoder in ("swin", "swin_from_scratch"):
        print("Using encoder:", args.encoder)
        if args.encoder == "swin_from_scratch":
            encoder = SwinModel(SwinConfig())  
        else:
            encoder = SwinModel.from_pretrained(
                "microsoft/swin-base-patch4-window7-224-in22k"
            )
        decoder_config = BertConfig(
            is_decoder=True,
            add_cross_attention=True,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        decoder = BertLMHeadModel(decoder_config)
        # full model -- swin
        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    else:
        trocr_model = "microsoft/" + args.encoder
        encoder = VisionEncoderDecoderModel.from_pretrained(trocr_model).encoder
    
        if args.decoder == "bert":
            decoder_config = BertConfig(
                is_decoder=True,
                add_cross_attention=True,
                vocab_size=len(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            decoder = BertLMHeadModel(decoder_config)
        
        elif args.decoder == "gpt2":
            decoder_config = GPT2Config(
                is_decoder=True,
                add_cross_attention=True,
                vocab_size=len(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
            )
            decoder = GPT2LMHeadModel(decoder_config)
        
        else:
            raise ValueError(f"unsupported decoder type: {args.decoder}")
        
        # full model 
        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # configs for the model
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    
    # setting beam search params
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 0
    model.config.num_beams = 5

    print(" Checking model configuration:")
    print(f"  decoder_start_token_id: {model.config.decoder_start_token_id}")
    print(f"  pad_token_id: {model.config.pad_token_id}")
    print(f"  eos_token_id: {model.config.eos_token_id}")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  max_length: {model.config.max_length}")
    print(f"  num_beams: {model.config.num_beams}")
    print(f"  no_repeat_ngram_size: {model.config.no_repeat_ngram_size}")
    print(f"  early_stopping: {model.config.early_stopping}\n")
    

    # small sample to check the model and tokenizre
    test_str = "बजारमा नयाँ पुस्तक आएको छ"
    encoded = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    print(f"  Test string: {test_str}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Tokens: {tokens}")

    # # eval steps  (preferable per epoch)
    # num_epochs = 20
    # total_steps = len(train_ds) // 8 * num_epochs
    # eval_steps = total_steps // num_epochs
    # print(f"Total training steps: {total_steps}, Eval steps: {eval_steps}")

    # training
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        # eval_steps= eval_steps, 
        logging_steps=100,
        warmup_steps=500,
        num_train_epochs=20, 
        learning_rate=3e-5, 
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        max_grad_norm=0.5,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True, 
        metric_for_best_model="cer", 
        greater_is_better=False,  
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[print_callback, early_stop_callback]
    )

    start_time = time.time()
    trainer.train()

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.utcfromtimestamp(elapsed_time).strftime('%H:%M:%S'))

    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # val set eval
    results_val = trainer.evaluate(val_ds)
    results_val["train_time"] = formatted_time
    results_val["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results_val["split"] = "val"

    # test set eval
    results_test = trainer.evaluate(test_ds)
    results_test["train_time"] = formatted_time
    results_test["num_parameters"] = results_val["num_parameters"]
    results_test["split"] = "test"

    # saving 
    csv_file = os.path.join(args.model_dir, f"{args.model_name}_log.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_val.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results_val)
        writer.writerow(results_test)

    print(" Results on test set:")
    print(results_test)

    wandb.finish()

    

# always make sure to define dataset_name (finetuning from scratch)
# always make sure to define dataset_name + finetune_from_model (finetuning from the pretrained model)
# oldNepaliSynthetic = pretraining dataset , nagari = finetuning dataset, oldNepali = main dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices = ['oldNepaliSynthetic_105k_vnoisy', 'nagari_original', 'nagari', 'oldNepali_original','oldNepali_aug16'], default="oldNepaliSynthetic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # model setup args
    parser.add_argument("--encoder", type=str, choices=["swin","swin_from_scratch", "trocr-small-handwritten", "trocr-base-handwritten", "trocr-large-handwritten"], default="trocr-base-handwritten")
    parser.add_argument("--decoder", type=str, choices=["bert", "gpt2"], default="bert")
    parser.add_argument("--tokenizer_type", type=str,  choices=["charBPE", "byteBPE", "sentencepieceBPE"], default="charBPE")
    parser.add_argument("--vocab_size", type=int, default=1000)
    
    # finetuning args   
    parser.add_argument("--finetune_from_model", type=str, default=None, help="Path to pretrained model to finetune from")
   
    args = parser.parse_args()

    # model name configurations
    if args.finetune_from_model:
        model_base = os.path.basename(args.finetune_from_model.strip("/"))
        args.model_name = f"{model_base}_finetuned_on_{args.dataset_name}"
    else: 
        args.model_name = f"{args.encoder}-{args.decoder.upper()}-{args.dataset_name}-{args.tokenizer_type}-{args.vocab_size}"
        
    args.model_dir = os.path.join("models/trained/", args.model_name)


    main(args)

