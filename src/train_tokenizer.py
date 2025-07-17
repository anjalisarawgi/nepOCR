import os 
from transformers import PreTrainedTokenizerFast
from tokenizers import CharBPETokenizer, SentencePieceBPETokenizer, ByteLevelBPETokenizer, BertWordPieceTokenizer

def train_tokenizer(corpus_path, tokenizer_type, vocab_size, output_dir):
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.readlines()

        if tokenizer_type == "charBPE":
            tokenizer_obj = CharBPETokenizer()
        elif tokenizer_type == "byteBPE":
            tokenizer_obj = ByteLevelBPETokenizer()
        elif tokenizer_type == "sentencepieceBPE":
            tokenizer_obj = SentencePieceBPETokenizer()
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        tokenizer_obj.train_from_iterator(corpus, vocab_size = vocab_size)
        os.makedirs(output_dir, exist_ok=True)
        tokenizer_obj.save(os.path.join(output_dir, "tokenizer.json"))

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = os.path.join(output_dir, "tokenizer.json"),
            pad_token = "[PAD]", 
            cls_token = "[CLS]", 
            eos_token = "[SEP]", 
            unk_token = "[UNK]"
        )

        tokenizer.save_pretrained(output_dir)
        return tokenizer
    
if __name__ == "__main__":
    
    corpus_path = "corpus/30k_nagari_oldNepali_v3.txt"
    tokenizer_type = "byteBPE"  
    vocab_size = 200

    tokenizer = train_tokenizer(
            corpus_path = corpus_path,
            tokenizer_type=tokenizer_type,
            vocab_size=vocab_size,
            output_dir=f"tokenizer/{tokenizer_type}_{vocab_size}"
    )
