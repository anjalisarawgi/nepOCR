import unicodedata
import pandas as pd
from collections import defaultdict

df_lemmas = pd.read_csv("lemmas.csv")
form_to_lemmas = defaultdict(set)

for _, row in df_lemmas.iterrows():
    form = unicodedata.normalize("NFC", row["word_form"].strip())
    lemma = unicodedata.normalize("NFC", row["lemma"].strip())
    form_to_lemmas[form].add(lemma)

form_to_shortest_lemma = {
    f: min(lems, key=len)
    for f, lems in form_to_lemmas.items()
}

MATRA_SIGNS = {'ा','ि','ी','ु','ू','े','ै','ो','ौ','ं','ः','ँ'}

def split_by_consonant_and_matra(text):
    text = unicodedata.normalize("NFC", text)
    chunks = []
    i = 0
    while i < len(text):
        char = text[i]
        next_char = text[i+1] if i+1 < len(text) else ''
        if next_char in MATRA_SIGNS:
            chunks.append(char + next_char)
            i += 2
        else:
            chunks.append(char)
            i += 1
    return chunks

def lemmatize_chunks_raw(text, max_chunk_len=15, min_match_len=2):
    text = unicodedata.normalize("NFC", text)
    lemmas = []
    matched_tokens = []
    i = 0
    while i < len(text):
        matched = False
        for span in range(min(max_chunk_len, len(text) - i), min_match_len - 1, -1):  # Enforce min length
            substring = text[i:i+span]
            if substring in form_to_shortest_lemma:
                lemmas.append(form_to_shortest_lemma[substring])
                matched_tokens.append(substring)
                i += span
                matched = True
                break
        if not matched:
            # Only fallback to 1-char if absolutely no match
            lemmas.append(text[i])
            i += 1
    return {
        "lemma": "".join(lemmas),
        "matched_tokens": matched_tokens,
        "matched_count": len(matched_tokens),
        "matched_char_count": sum(len(token) for token in matched_tokens),
        "total_chars": len(text),
        "match_ratio": sum(len(token) for token in matched_tokens) / len(text) if text else 0.0
    }

sentence = "पिकामगर्याका101घोडालाईचादिकाहैकल्"
result = lemmatize_chunks_raw(sentence)

print("🔍 Original:", sentence)
print("🧩 Chunks :", split_by_consonant_and_matra(sentence))
print("📚 Lemma  :", result["lemma"])
print("✅ Matched Tokens:", result["matched_tokens"])
print(f"📈 Match Ratio: {result['match_ratio']:.2f} ({result['matched_char_count']}/{result['total_chars']} chars)")