import re

def clean_line(text):
    # Remove English letters and digits
    text = re.sub(r'[a-zA-Z0-9]', '', text)

    # Remove unwanted symbols (everything not Devanagari, whitespace, or selected punctuations)
    # text = re.sub(r'[^\u0900-\u097F\s।॥,:\-–]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


input_file = 'corpus/oldNepaliSynthetic10k_nagari_oldNepali.txt'
output_file = 'corpus/oldNepaliSynthetic10k_nagari_oldNepali_clean.txt'
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Clean each line individually
cleaned_lines = [clean_line(line) for line in lines if clean_line(line)]

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned_lines))