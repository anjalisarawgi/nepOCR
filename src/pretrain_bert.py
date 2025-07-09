from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BertLMHeadModel
from datasets import load_dataset

from src.utils.tokenizer import train_tokenizer
from transformers import PreTrainedTokenizerFast
import argparse
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/char_BPE")

config = BertConfig(
    is_decoder = True,
    add_cross_attention=True,
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
decoder = BertLMHeadModel(config)

dataset = load_dataset("text", data_files="corpus/nepali_texts_210k.txt")  # contains raw oldNepali texts extracted from books (size ~ 210k)

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


args = TrainingArguments(
    output_dir="models/pretrained_bert_decoder",
    per_device_train_batch_size=256,
    evaluation_strategy="no",
    num_train_epochs=10,
    learning_rate=3e-5,
    logging_steps=500,
    report_to="wandb", 
    save_strategy="no",
     weight_decay=0.01,
)

trainer = Trainer(
    model=decoder,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=data_collator, 
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("models/pretrained_bert_decoder")
tokenizer.save_pretrained("models/pretrained_bert_decoder")