from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")  # or your custom one

# Your input string
text = "कम्यान्डरइनचिफ्भीमसेषापासाधलाउन्यासंधियारथाघरगणपतिपंथगोकुलषनालजयदेवपंषउगोलना"

# Tokenize and check length
tokens = tokenizer(text, return_tensors="pt")
input_len = tokens.input_ids.shape[1]

# Model max length
max_len = tokenizer.model_max_length

print(f"Input length: {input_len}")
print(f"Model max length: {max_len}")
print(f"✅ Within limit? {input_len <= max_len}")