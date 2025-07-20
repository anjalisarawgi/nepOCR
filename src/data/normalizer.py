import json
import unicodedata
import re
from collections import defaultdict

# Devanagari diacritics and marks
VIRAMA = "\u094D"
NUKTA = "\u093C"
MACRON_PATTERN = re.compile(r"\u0304+")

# Patterns
_PATTERNS = {
    "removed_invis_chars": re.compile(r"[\u200B\u200C\u200D\u00AD]"),
    "bullet_to_dot": re.compile(r"[·•‧∙]"),
    "removed_single_quotes": re.compile(r"[`'‘’]"),
    "removed_whitespace": re.compile(r"\s+", re.UNICODE),
    "noisy_char": re.compile(r"[\u0300\u0301\u034F]")
}

def normalize_text(s, counter):
    s = unicodedata.normalize("NFKC", s)

    for key, pattern in _PATTERNS.items():
        s, n = pattern.subn(
            {
                "bullet_to_dot": ".",
                "removed_single_quotes": "",
                "removed_invis_chars": "",
                "removed_whitespace": "", 
                "noisy_char": ""
            }[key], s
        )
        if n:
            counter[key] += n

    if "||" in s:
        s = s.replace("||", "॥")
        counter["pipe_to_double_danda"] += 1
    if "|" in s:
        s = s.replace("|", "।")
        counter["pipe_to_danda"] += 1

    for char in "()":
        if char in s:
            counter["removed_parens"] += s.count(char)
            s = s.replace(char, "")


    # v6 - enable
    if "\\" in s:
        counter["removed_backslashes"] += s.count("\\")
        s = s.replace("\\", "")
    

    if NUKTA in s:
        counter["removed_nukta"] += s.count(NUKTA)
        s = s.replace(NUKTA, "")

    matches = MACRON_PATTERN.findall(s)
    if matches:
        counter["removed_macrons"] += sum(len(m) for m in matches)
        s = MACRON_PATTERN.sub("-", s)


    s = s.strip()
    return s

def main():
    input_path = "data/oldNepali/processed/raw_labels/labels_train_raw.json"
    output_path = "data/oldNepali/processed/cleaned_labels/labels_train.json"

    with open(input_path, encoding="utf8") as f:
        labels = json.load(f)

    counter = defaultdict(int)
    for rec in labels:
        rec["text"] = normalize_text(rec["text"], counter)

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"\n Summary of changes:")
    for k, v in sorted(counter.items()):
        print(f"  {k:<30s}: {v}")

if __name__ == "__main__":
    main() 