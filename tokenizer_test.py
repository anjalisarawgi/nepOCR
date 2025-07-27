# from transformers import AutoTokenizer

# # Load your tokenizer
# tokenizer = AutoTokenizer.from_pretrained("tokenizer/byteBPE_500")

# # Your input string
# text = "त­यदुनाथ­पंडित­अर्ज्याल­पानि­हालन्या­श्रीजनरल­भीमसेन­थापा­साध­लाउन्या­सधियार­मोदनाथ­अर्ज्याल­मेघनाथ­पाडे­व्राह्मण­टोलको­अमित­सिं­डगोल­"

# # Tokenize and check length
# tokens = tokenizer(text, return_tensors="pt")
# input_len = tokens.input_ids.shape[1]

# # Model max length
# max_len = tokenizer.model_max_length

# print(f"Input length: {input_len}")
# print(f"Model max length: {max_len}")
# print(f"✅ Within limit? {input_len <= max_len}")

import unicodedata

s = "श्रीदुर्गासहाय­\\"
for ch in s:
    print(
        repr(ch).ljust(6),
        hex(ord(ch)).ljust(8),
        unicodedata.name(ch, "<unknown>")
    )