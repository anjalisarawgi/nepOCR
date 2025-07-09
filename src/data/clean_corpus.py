import re

def clean_line(text):
    # Remove English letters and digits
    text = re.sub(r'[a-zA-Z0-9$%#!@&?]', '', text)

    return text.strip()


input_file = 'corpus/oldNepaliSynthetic_nagari_oldNepali.txt'
output_file = 'corpus/oldNepaliCorpus.txt'
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Clean each line individually
cleaned_lines = [clean_line(line) for line in lines if clean_line(line)]

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned_lines))