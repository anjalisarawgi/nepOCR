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
    PreTrainedTokenizerFast
)
from utils.data import load_dataset, preprocess_dataset, OCRTorchDataset, collate_fn
from utils.tokenizer import train_tokenizer
from utils.callbacks import PrintPredictionsCallback
from utils.metrics import compute_metrics
# from utils.seed import set_all_seeds

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    set_seed(42)
    # set_all_seeds(args.seed)
    wandb.init(project="nepOCR-logs", name = args.model_name)

    # dataset
    if args.dataset_name == "oldNepaliSynthetic":
        dataset = load_dataset("json", data_files="data/oldNepaliSynthetic/10k/labels_processed_new.json")
        split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    elif args.dataset_name == "nagari":
        train_dataset = load_dataset("data/nagari/augmented3/train/labels_processed_new.json")
        eval_dataset = load_dataset("data/nagari/augmented3/test/labels_processed_new.json")
    elif args.dataset_name == "oldNepali":
        train_dataset = load_dataset("data/oldNepali/augmented3/train/labels_processed.json")
        eval_dataset = load_dataset("data/oldNepali/augmented3/test/labels_processed.json")
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")

    # slice for debugging
    # train_dataset = train_dataset.select(range(200))
    # eval_dataset = eval_dataset.select(range(50))

    # tokenizer
    if args.finetune_from_model: # if finetuning from a pretrained model, load the tokenizer from the model (esp for nagari and oldNepali)
        model_path = args.finetune_from_model
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    else:
        print(f"Training tokenizer from corpus: corpus/oldNepaliSynthetic_nagari_oldNepali.txt")
        tokenizer = train_tokenizer(
            corpus_path="corpus/oldNepaliSynthetic_nagari_oldNepali.txt",
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            output_dir=f"tokenizer/{args.tokenizer_type}_{args.vocab_size}",
        )

    print("\n Debugging and checking the tokenizer:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})\n")

    # processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    feature_extractor = processor.feature_extractor


    train_imgs, train_lbls = preprocess_dataset(train_dataset, tokenizer, feature_extractor, max_length=100)
    eval_imgs, eval_lbls = preprocess_dataset(eval_dataset, tokenizer, feature_extractor, max_length=100)

    train_ds = OCRTorchDataset(train_imgs, train_lbls)
    eval_ds = OCRTorchDataset(eval_imgs, eval_lbls)


    # callback initialized for debugging and inference 
    sample_batch = [eval_ds[i] for i in range(5)]
    print_callback = PrintPredictionsCallback(sample_batch, tokenizer, print_every=100)


    # model    
    if args.finetune_from_model:
        print(f"Finetuning from: {args.finetune_from_model}")
        model = VisionEncoderDecoderModel.from_pretrained(args.finetune_from_model)
    else:
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
    model.config.vocab_size = model.config.decoder.vocab_size

    
    # setting beam search params
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = 100
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 10
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
    # epochs = 15
    # batch_size = 8
    
    # # eval_steps  = len(train_ds) // args.batch_size # does it once every epich
    # eval_steps = max(1, (len(train_ds) * epochs ) // ( batch_size * 6)) # evaluates 6 times every run
    # total_steps = (len(train_ds) * epochs) // batch_size
    # warmup_steps =  max(1, int(0.1 * total_steps))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=1000, 
        logging_steps=100,
        warmup_steps=500,
        num_train_epochs=15,
        learning_rate=3e-5,
        weight_decay=0.01,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"],
        max_grad_norm=0.5,
        save_strategy="no",
        # gradient_accumulation_steps=2 if torch.cuda.is_available() else 1,
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

# always make sure to define dataset_name (finetuning from scratch)
# always make sure to define dataset_name + finetune_from_model (finetuning from the pretrained model)
# oldNepaliSynthetic = pretraining dataset , nagari = finetuning dataset, oldNepali = main dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices = ['nagari', 'oldNepaliSynthetic', 'oldNepali'], default="oldNepaliSynthetic")

    # model setup args
    parser.add_argument("--encoder", type=str, choices=["trocr"], default="trocr")
    parser.add_argument("--decoder", type=str, choices=["bert"], default="bert")
    parser.add_argument("--tokenizer_type", type=str, choices=["char", "sbpe"], default="char")
    parser.add_argument("--vocab_size", type=int, default=1000)
    
    # finetuning args   
    parser.add_argument("--finetune_from_model", type=str, default=None, help="Path to pretrained model to finetune from")
   
    args = parser.parse_args()

    # model name logic
    if args.finetune_from_model:
        model_base = os.path.basename(args.finetune_from_model.strip("/"))
        args.model_name = f"{model_base}_finetuned_on_{args.dataset_name}"
    else:
        args.model_name = f"{args.encoder}-{args.decoder.upper()}-{args.dataset_name}-{args.tokenizer_type}-{args.vocab_size}"
    args.model_dir = os.path.join("models", args.model_name)

    main(args)

