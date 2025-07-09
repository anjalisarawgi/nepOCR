import itertools
import json

encoders = [
            # 'trocr-small-hanwritten', 
            'trocr-base-hanwritten', 
            # 'trocr-large-hanwritten'
            ]

decoders = [
    # 'roberta', 
    'bert',
    'gpt2',
    # 't5',
    # 'byt5',
    # 'mbart',
]

tokenizers = [
    'ByteLevelBPETokenizer',
    'CharBPETokenizer',
    # 'SentencePieceBPETokenizer',
    # 'BertWordPieceTokenizer',
    # 'unigram'
]

vocab_sizes = [500, 1000, 5000]

# Make all combinations
all_experiments = list(itertools.product(encoders, decoders, tokenizers, vocab_sizes))

# Save one-line per item
with open('experiment_configurations.jsonl', 'w') as f:
    for (enc, dec, tok, vocab) in all_experiments:
        item = {
            "encoder": enc,
            "decoder": dec,
            "tokenizer": tok,
            "vocab_size": vocab
        }
        json.dump(item, f, separators=(',', ':'))
        f.write('\n')  # Write each object in one line

print(f"Total experiments: {len(all_experiments)}")