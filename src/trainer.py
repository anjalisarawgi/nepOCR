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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

start_time = time.time()



def main(args):
    set_seed(args.seed)
    wandb.init(project="nepOCR-logs", name = args.model_name)

    # dataset
    if args.dataset_name == "oldNepaliSynthetic":
        dataset = load_dataset("data/oldNepaliSynthetic10k/labels.json")
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "oldNepaliSynthetic30k":
        dataset = load_dataset("data/oldNepaliSynthetic30k/labels_processed.json")
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
    elif args.dataset_name == "oldNepaliSynthetic_105k_vnoisy":
        dataset = load_dataset("data/oldNepaliSynthetic_105k_vnoisy/labels_processed.json")
        split_dataset = dataset.train_test_split(train_size=100000, seed=args.seed)
        train_dataset = split_dataset["train"]
        full_eval_dataset = split_dataset["test"]
        eval_dataset = full_eval_dataset.select(range(0, 2500))
        val_dataset = eval_dataset
        test_dataset = full_eval_dataset.select(range(2500, len(full_eval_dataset)))
    # elif args.dataset_name == "nagari":
    #     train_dataset = load_dataset("data/nagari/augmented3/train/labels_cleaned_v2.json")
    #     full_eval_dataset = load_dataset("data/nagari/augmented3/test/labels_cleaned_v2.json")
    #     split = full_eval_dataset.train_test_split(train_size=0.5, shuffle=False)
    #     val_dataset = split["train"]
    #     test_dataset = split["test"]
    elif args.dataset_name == "nagari":
        train_dataset = load_dataset("data/nagari/augmented4/train/labels_train.json")
        val_dataset = load_dataset("data/nagari/original/train/labels_val.json")
        test_dataset = load_dataset("data/nagari/augmented4/test/labels_test.json")
    elif args.dataset_name == "nagari_original":
        train_dataset = load_dataset("data/nagari/original/train/labels.json")
        eval_dataset = load_dataset("data/nagari/original/test/labels.json")
    elif args.dataset_name == "oldNepali_original":
        train_dataset = load_dataset("data/oldNepali/original/train/labels.json")
        eval_dataset = load_dataset("data/oldNepali/original/test/labels.json")
    elif args.dataset_name == "oldNepaliDataset3":
        train_dataset = load_dataset("data/oldNepaliDataset3/labels_train_processed.json")
        eval_dataset = load_dataset("data/oldNepaliDataset/labels_test_processed.json")
    elif args.dataset_name == "oldNepaliDataset_new3":
        train_dataset = load_dataset("data/oldNepaliDataset_new3/labels_train_processed_cleaned_newv9.json") # labels_train_processed_cleaned_new ---- labels_train_processed_cleaned_newv4
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv9.json")
    elif args.dataset_name == "oldNepaliDataset_new5":
        train_dataset = load_dataset("data/oldNepaliDataset_new5/labels_train_processed_cleaned_newv12.json") # labels_train_processed_cleaned_newv9 ----- labels_train_processed_cleaned
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
    elif args.dataset_name == "oldNepaliDataset_new8":
        train_dataset = load_dataset("data/oldNepaliDataset_new8/labels_train_processed_cleaned_newv12.json") # labels_train_processed_cleaned_newv9 ----- labels_train_processed_cleaned
        eval_dataset = load_dataset("splits/labels_val.json")
        val_dataset = eval_dataset
        test_dataset = load_dataset("splits/labels_test.json")
        # eval_dataset = eval_dataset.shuffle(seed=42)
        # test_dataset = eval_dataset.select(range(150))
        # val_dataset = eval_dataset.select(range(150, len(eval_dataset)))
        # eval_dataset = test_dataset
        # print("length of eval_dataset:", len(eval_dataset))
        # print("length of val_dataset:", len(val_dataset))
    elif args.dataset_name == "oldNepaliDataset_new10_v2":
        train_dataset = load_dataset("data/oldNepaliDataset_new10_v2/labels_train_processed_cleaned_newv12_shuffled.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
    elif args.dataset_name == "oldNepaliDataset_binarized3":
        train_dataset = load_dataset("data/oldNepaliDataset_binarized3/labels_train_processed_cleaned.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_binarized/labels_test_processed_cleaned.json")
    elif args.dataset_name == "oldNepaliSynthetic_difficult_2k_clean_vnoisy":
        dataset = load_dataset("data/oldNepaliSynthetic_difficult_2k_clean_vnoisy/labels.json")
        split_dataset = dataset.train_test_split(train_size=1600, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "new_synthetic_vnoisy":
        dataset = load_dataset("data/new_synthetic_vnoisy/labels.json")
        split_dataset = dataset.train_test_split(test_size = 0.1, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        eval_dataset = eval_dataset.select(range(1000))
    elif args.dataset_name == "top5_confusion_lines_1000_vnoisy":
        dataset = load_dataset("data/top5_confusion_lines_1000_vnoisy/labels.json")
        split_dataset = dataset.train_test_split(train_size = 900, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "handwritten_aug8":
        train_dataset = load_dataset("data/handwritten/labels_handwritten_v12_aug8_shuffled.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
    elif args.dataset_name == "char_htr_aug8":
        train_dataset = load_dataset("data/char_htr_aug8/labels_train_processed.json")
        eval_dataset = load_dataset("data/char_htr_aug8/labels_test_processed.json")
    elif args.dataset_name == "char_htr_aug8_nosplit":
        dataset = load_dataset("data/char_htr_aug8_nosplit/labels.json")
        split_dataset = dataset.train_test_split(train_size = 0.99, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "oldNepaliDataset_new8_chava":
        train_dataset = load_dataset("data/oldNepaliDataset_new8/labels_train_cha_v12.json") # labels_train_processed_cleaned_newv9 ----- labels_train_processed_cleaned
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
    elif args.dataset_name == "dev_char_kaggle":
        dataset = load_dataset("data/dev_char_kaggle/labels.json")
        split_dataset = dataset.train_test_split(train_size = 12000, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "oldNepaliDataset_new12":
        train_dataset = load_dataset("data/oldNepaliDataset_new12/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
    elif args.dataset_name == "oldNepaliDataset_new16":
        train_dataset = load_dataset("data/oldNepaliDataset_new16/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
        test_dataset = eval_dataset
        val_dataset = eval_dataset 
        # eval_dataset = eval_dataset.shuffle(seed=41)
        # test_dataset = eval_dataset.select(range(150))
        # val_dataset = eval_dataset.select(range(150, len(eval_dataset)))
        # eval_dataset = test_dataset
        # print("length of eval_dataset:", len(eval_dataset))
        # print("length of val_dataset:", len(val_dataset))
    elif args.dataset_name == "oldNepaliDataset_new18":
        train_dataset = load_dataset("data/oldNepaliDataset_new18/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
        test_dataset = eval_dataset
        val_dataset = eval_dataset
    elif args.dataset_name == "oldNepaliDataset_new8_new":
        train_dataset = load_dataset("data/oldNepaliDataset_new8_new/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/new/test.json")
        # test_dataset = eval_dataset.select(range(150))
        # val_dataset = eval_dataset.select(range(150, len(eval_dataset)))
        test_dataset = eval_dataset
        val_dataset = eval_dataset
        print("length of eval_dataset:", len(eval_dataset))
        print("length of val_dataset:", len(val_dataset))
    elif args.dataset_name == "handwriiten_aug5":
        train_dataset = load_dataset("data/handwriiten_aug5/labels_train.json")
        eval_dataset = train_dataset
        test_dataset = eval_dataset
        val_dataset = eval_dataset
    elif args.dataset_name =="oldNepaliDataset_new8_synth15":
        train_dataset = load_dataset("data/oldNepaliDataset_new8_synth15/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new/labels_test_processed_cleaned_newv12.json")
        test_dataset = eval_dataset
        val_dataset = eval_dataset
    elif args.dataset_name == "oldNepaliDataset_new8_val":
        train_dataset = load_dataset("data/oldNepaliDataset_new8_val/labels_train.json")
        eval_dataset = load_dataset("splits/labels_test.json")
        val_dataset=eval_dataset
        test_dataset = eval_dataset
    elif args.dataset_name == "oldNepaliDataset_new_42":
        train_dataset = load_dataset("data/oldNepaliDataset_new_42_aug16/labels_train.json")
        eval_dataset = load_dataset("data/oldNepaliDataset_new_42/labels_test.json")
        test_dataset = eval_dataset
        val_dataset = load_dataset("data/oldNepaliDataset_new_42/labels_val.json")
    elif args.dataset_name == "oldNepali_aug16":
        train_dataset = load_dataset("data/oldNepali_aug16/labels_train.json")
        eval_dataset = load_dataset("data/oldNepali/processed/labels_test.json")
        test_dataset = eval_dataset
        val_dataset = load_dataset("data/oldNepali/processed/labels_val.json")
    elif args.dataset_name == "oldNepali":
        train_dataset = load_dataset("data/oldNepali/processed/labels_train_raw.json")
        eval_dataset = load_dataset("data/oldNepali/processed/labels_test_raw.json")
        test_dataset = eval_dataset
        val_dataset = load_dataset("data/oldNepali/processed/labels_val_raw.json")
    
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")


    # slice for debugging
    # train_dataset = train_dataset.select(range(200))
    # eval_dataset = eval_dataset.select(range(50))

    print(f"Train dataset size: {len(train_dataset)} examples")
    print(f"Eval  dataset size: {len(val_dataset)} examples\n")
    print(f"Test  dataset size: {len(test_dataset)} examples")

    # tokenizer
    if args.use_full_trocr:
        model_path = "microsoft/" + args.full_torcr_model_name
        print("Using full TrOCR tokenizer and processor from:", model_path)
        processor = TrOCRProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        feature_extractor = processor.feature_extractor 
    else:
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
    if args.use_full_trocr:
        model_path = "microsoft/" + args.full_torcr_model_name
        print("Loading processor from: ", model_path)
    else:
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


    # model    
    if args.use_full_trocr:
        print("Using full TrOCR model (encoder + decoder)")
        full_trocr_model = "microsoft/" + args.full_torcr_model_name
        model = VisionEncoderDecoderModel.from_pretrained(full_trocr_model)
    else:
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

    # training
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
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
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[print_callback]
    )
    start_time = time.time()
    trainer.train()

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(datetime.utcfromtimestamp(elapsed_time).strftime('%H:%M:%S'))
    


    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # --- Timing and Evaluation ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = str(datetime.utcfromtimestamp(elapsed_time).strftime('%H:%M:%S'))

    # --- Final Evaluation ---
    results = trainer.evaluate(test_ds)
    results["train_time"] = formatted_time
    results["model_name"] = args.model_name
    results["dataset_name"] = args.dataset_name
    results["finetune_from_model"] = args.finetune_from_model or "None"
    results["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["beam_size"] = model.config.num_beams
    results["max_length"] = model.config.max_length
    results["tokenizer_type"] = args.tokenizer_type
    results["vocab_size"] = args.vocab_size

    # --- CSV Log Path ---
    csv_file = os.path.join(args.model_dir, f"{args.model_name}_log.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # --- Write to CSV ---
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    # --- Print Summary ---
    print("\n📊 Training Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


    print("Final Evaluation:", trainer.evaluate(test_ds))
    wandb.finish()

    

# always make sure to define dataset_name (finetuning from scratch)
# always make sure to define dataset_name + finetune_from_model (finetuning from the pretrained model)
# oldNepaliSynthetic = pretraining dataset , nagari = finetuning dataset, oldNepali = main dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices = ['nagari', 'char_htr_aug8_nosplit', 'oldNepali','oldNepaliDataset_new_42', 'handwriiten_aug5', 'dev_char_kaggle', 'oldNepaliDataset_new8_chava', 'char_htr_aug8', 'oldNepaliSynthetic', 'oldNepali', 'top5_confusion_lines_1000_vnoisy', 'oldNepaliSynthetic_difficult_2k_clean_vnoisy', 'new_synthetic_vnoisy', 'oldNepaliDataset_new7', 'oldNepaliDataset_binarized3', 'oldNepaliSynthetic30k', 'oldNepaliSynthetic_105k', 'nagari_original', 'oldNepali_original', 'oldNepaliSynthetic_105k_vnoisy', 'oldNepaliDataset', 'oldNepaliDataset3', 'oldNepaliDataset_new3', 'oldNepaliDataset_new5', 'oldNepaliDataset_new8', "oldNepaliDataset_new10_v2", "handwritten_aug8", 'oldNepaliDataset_new12', 'oldNepaliDataset_new16', 'oldNepaliDataset_new8_new', 'oldNepaliDataset_new18', 'oldNepaliDataset_new8_synth15', 'oldNepaliDataset_new8_val', 'oldNepaliDataset_new_v2_16', 'oldNepali_aug16' ], default="oldNepaliSynthetic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # model setup args
    parser.add_argument("--encoder", type=str, choices=["swin","swin_from_scratch", "trocr-small-handwritten", "trocr-base-handwritten", "trocr-large-handwritten"], default="trocr-base-handwritten")
    parser.add_argument("--decoder", type=str, choices=["bert", "gpt2"], default="bert")
    parser.add_argument("--tokenizer_type", type=str,  choices=["charBPE", "byteBPE", "sentencepieceBPE"], default="charBPE")
    parser.add_argument("--vocab_size", type=int, default=1000)
    
    # finetuning args   
    parser.add_argument("--finetune_from_model", type=str, default=None, help="Path to pretrained model to finetune from")
   
   # args if you want to use base TrOCR (trocr as it is)
    parser.add_argument("--use_full_trocr", action="store_true", help="set this if you want to use TrOCR as it is (both encoder and decoder).")
    parser.add_argument("--full_torcr_model_name", type=str, choices=["trocr-small-handwritten", "trocr-base-handwritten", "trocr-large-handwritten"], default="trocr-base-handwritten",help="Full TrOCR model (only used if --use_full_trocr is set)")


    args = parser.parse_args()

    # model name configurations
    if args.finetune_from_model:
        model_base = os.path.basename(args.finetune_from_model.strip("/"))
        args.model_name = f"{model_base}_finetuned_on_{args.dataset_name}"
    else: 
        if args.use_full_trocr:
            args.model_name = f"full-trocr-{args.dataset_name}"
        else:
            args.model_name = f"{args.encoder}-{args.decoder.upper()}-{args.dataset_name}-{args.tokenizer_type}-{args.vocab_size}"
        
    args.model_dir = os.path.join("models/trained/", args.model_name)


    main(args)

