# import json
# import unicodedata
# import re
# from collections import defaultdict

# # Devanagari diacritics and marks
# VIRAMA = "\u094D"
# NUKTA = "\u093C"
# MACRON_PATTERN = re.compile(r"\u0304+")

# # Patterns
# _PATTERNS = {
#     "removed_invis_chars": re.compile(r"[\u200B\u200C\u200D\u00AD]"),
#     "bullet_to_dot": re.compile(r"[·•‧∙]"),
#     "removed_single_quotes": re.compile(r"[`'‘’]"),
#     "removed_whitespace": re.compile(r"\s+", re.UNICODE),
#     "noisy_char": re.compile(r"[\u0300\u0301\u034F]")
# }

# def normalize_text(s, counter):
#     s = unicodedata.normalize("NFKC", s)

#     for key, pattern in _PATTERNS.items():
#         s, n = pattern.subn(
#             {
#                 "bullet_to_dot": ".",
#                 "removed_single_quotes": "",
#                 "removed_invis_chars": "",
#                 "removed_whitespace": "", 
#                 "noisy_char": ""
#             }[key], s
#         )
#         if n:
#             counter[key] += n

#     if "||" in s:
#         s = s.replace("||", "॥")
#         counter["pipe_to_double_danda"] += 1
#     if "|" in s:
#         s = s.replace("|", "।")
#         counter["pipe_to_danda"] += 1

#     for char in "()":
#         if char in s:
#             counter["removed_parens"] += s.count(char)
#             s = s.replace(char, "")


#     # # v6 - enable
#     if "\\" in s:
#         counter["removed_backslashes"] += s.count("\\")
#         s = s.replace("\\", "")


#     if NUKTA in s:
#         counter["removed_nukta"] += s.count(NUKTA)
#         s = s.replace(NUKTA, "")

#     matches = MACRON_PATTERN.findall(s)
#     if matches:
#         counter["removed_macrons"] += sum(len(m) for m in matches)
#         s = MACRON_PATTERN.sub("", s)


#     s = s.strip()
#     return s

# def main():
#     input_path = "data/oldNepali/processed/raw_labels/labels_train_raw.json"
#     output_path = "data/oldNepali/processed/cleaned_labels/labels_train.json"

#     with open(input_path, encoding="utf8") as f:
#         labels = json.load(f)

#     counter = defaultdict(int)
#     for rec in labels:
#         rec["text"] = normalize_text(rec["text"], counter)

#     with open(output_path, "w", encoding="utf8") as f:
#         json.dump(labels, f, ensure_ascii=False, indent=2)

#     print(f"\n Summary of changes:")
#     for k, v in sorted(counter.items()):
#         print(f"  {k:<30s}: {v}")

# if __name__ == "__main__":
#     main() 

import json
import unicodedata
import re


VIRAMA = "\u094D" 
_INVIS       = re.compile(r"[\u200B-\u200D\u00AD]")    
_DASH        = re.compile(r"[–—−]")          
_BULLET      = re.compile(r"[·•‧∙]")         
_MULTIDOT    = re.compile(r"\.{3,}")         
_WHITESPACE  = re.compile(r"[ \t\u00A0]+")   
_QUOTES        = re.compile(r'[“”]')         
_STRIP_SINGLE  = re.compile(r"[`'‘’]")   
_NUKTA = re.compile(r"\u093C") # NUKTA (dot below) → nothing


def remove_combining_with_logging(s):
    removed = [c for c in s if unicodedata.combining(c) and c != VIRAMA]
    if removed and VIRAMA not in s:
        print("⚠️ Removed combining characters (no Virama found):")
        for c in removed:
            print(f"  {repr(c)} | U+{ord(c):04X} | {unicodedata.name(c, 'UNKNOWN')} | combining={unicodedata.combining(c)}")
    return "".join(c for c in s if not (unicodedata.combining(c) and c != VIRAMA))


def normalize_text(s):
    s = unicodedata.normalize("NFKC", s)
    s = _INVIS.sub("", s)
    s = _DASH.sub("-", s)       
    s = _BULLET.sub(".", s)     
    s = _MULTIDOT.sub("..", s)
    s = s.replace("|", "।")
    s = s.replace("||", "॥")
    s = s.replace("(", "").replace(")", "")
    s = s.replace('\"', "।")
    s = _WHITESPACE.sub(" ", s).strip()
    s = _QUOTES.sub('"', s)
    s = _STRIP_SINGLE.sub("", s)
    # s = "".join(c for c in s if not unicodedata.combining(c)) 
    s = remove_combining_with_logging(s) # removes all combining characters except for VIRAMA
    # s = s.replace("¯", "") # remove upper dash
    s = s.replace(" ", "").replace("\u00A0", "").replace("\t", "") # remove all spaces
    s = _NUKTA.sub("", s) # remove NUKTA
    # s = s.replace("॥", "..")
    return s

def main():
    with open("data/oldNepali/processed/raw_labels/labels_test_raw.json", encoding="utf8") as f:
        labels = json.load(f)

    cleaned = []
    for rec in labels:
        orig = rec["text"]
        norm = normalize_text(orig)
        rec["text"] = norm        
        cleaned.append(rec)

    out_path = "data/oldNepali/processed/processed_labels/labels_test_v2.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Wrote cleaned labels to: {out_path}\n")

    print(f"{'ORIGINAL':40s} ↦ NORMALIZED")
    print("-"*80)
    for rec in labels[:5]:
        before = rec["text"] 
        after  = normalize_text(before)
        print(f"{before!r:40s} ↦ {after!r}")

if __name__ == "__main__":
    main()