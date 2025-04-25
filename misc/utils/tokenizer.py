import os 
from transformers import PreTrainedTokenizerFast
from tokenizers import CharBPETokenizer, SentencePieceBPETokenizer

def train_tokenizer(corpus_path, tokenizer_type, vocab_size, output_dir):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.readlines()

        tokenizer_obj = (
            CharBPETokenizer() if tokenizer_type == "char" else SentencePieceBPETokenizer()
        )

        tokenizer_obj.train_from_iterator(corpus, vocab_size = vocab_size)
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_obj.save(os.path.join(output_dir, "tokenizer.json"))

        return PreTrainedTokenizerFast(
            tokenizer_file = os.path.join(output_dir, "tokenizer.json"),
            pad_token = "[PAD]", 
            cls_token = "[CLS]", 
            eos_token = "[SEP]", 
            unk_token = "[UNK]"
        )
