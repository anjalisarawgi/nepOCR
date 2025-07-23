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

def normalize_text(s, counter, line_counter):
    s = unicodedata.normalize("NFKC", s)

    for key, pattern in _PATTERNS.items():
        s_new, n = pattern.subn({
            "bullet_to_dot": ".",
            "removed_single_quotes": "",
            "removed_invis_chars": "",
            "removed_whitespace": "", 
            "noisy_char": ""
        }[key], s)
        if n > 0:
            counter[key] += n
            line_counter[key] += 1
            s = s_new

    if "||" in s:
        s = s.replace("||", "॥")
        counter["pipe_to_double_danda"] += 1
        line_counter["pipe_to_double_danda"] += 1
    if "|" in s:
        s = s.replace("|", "।")
        counter["pipe_to_danda"] += 1
        line_counter["pipe_to_danda"] += 1

    for char in "()":
        if char in s:
            count = s.count(char)
            counter["removed_parens"] += count
            line_counter["removed_parens"] += 1
            s = s.replace(char, "")

    # Optional backslash removal (uncomment if needed)
    # if "\\" in s:
    #     count = s.count("\\")
    #     counter["removed_backslashes"] += count
    #     line_counter["removed_backslashes"] += 1
    #     s = s.replace("\\", "")

    if NUKTA in s:
        count = s.count(NUKTA)
        s = s.replace(NUKTA, "")
        counter["removed_nukta"] += count
        line_counter["removed_nukta"] += 1

    matches = MACRON_PATTERN.findall(s)
    if matches:
        total_len = sum(len(m) for m in matches)
        s = MACRON_PATTERN.sub("", s)
        counter["removed_macrons"] += total_len
        line_counter["removed_macrons"] += 1

    if "\u0310" in s:
        count = s.count("\u0310")
        s = s.replace("\u0310", "\u0901")
        counter["replaced_fake_candrabindu"] += count
        line_counter["replaced_fake_candrabindu"] += 1

    # s = unicodedata.normalize("NFKC", s)
    # s = normalize_digits(s, counter, line_counter, to="latin")

    s = s.strip()
    return s

def main():
    input_path = "data/oldNepali_fullset/labels_fixed.json"
    output_path = "data/oldNepali_fullset/labels_normalized.json"
    
    with open(input_path, encoding="utf8") as f:
        labels = json.load(f)

    total_lines = len(labels)
    counter = defaultdict(int)
    line_counter = defaultdict(int)
    total_lines_changed = 0
    for rec in labels:
        original = rec["text"]
        counter["total_chars_before"] += len(original)
        cleaned = normalize_text(original, counter, line_counter)
        counter["total_chars_after"] += len(cleaned)
        rec["text"] = cleaned

        if original != cleaned:
            total_lines_changed += 1
            

    with open(output_path, "w", encoding="utf8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    removed = counter["total_chars_before"] - counter["total_chars_after"]
    print(f"\n Summary of changes:")
    keys = sorted(set(counter) | set(line_counter))
    for k in keys:
        if k in ["total_chars_before", "total_chars_after"]:
            print(f"  {k:<30s}: {counter[k]}")
        else:
            print(f"  {k:<30s}: {counter[k]:<6d}   (lines affected: {line_counter[k]})")
    # Only include keys that represent a specific transformation
    total_chars_affected = sum(
        count for key, count in counter.items()
        if key not in {"total_chars_before", "total_chars_after"}
    )
    print(f"\n  Total characters affected             : {total_chars_affected}")
    print(f"\n  Characters removed during normalization: {removed}")
    print(f"\n Summary of changes:")
    print(f"  total_lines_processed           : {total_lines}")
    print(f"  total_lines_changed             : {total_lines_changed}")

if __name__ == "__main__":
    main()


# import json
# import unicodedata
# import re


# VIRAMA = "\u094D" 
# _INVIS       = re.compile(r"[\u200B-\u200D\u00AD]")    
# _DASH        = re.compile(r"[–—−]")          
# _BULLET      = re.compile(r"[·•‧∙]")         
# _MULTIDOT    = re.compile(r"\.{3,}")         
# _WHITESPACE  = re.compile(r"[ \t\u00A0]+")   
# _QUOTES        = re.compile(r'[“”]')         
# _STRIP_SINGLE  = re.compile(r"[`'‘’]")   
# _NUKTA = re.compile(r"\u093C") # NUKTA (dot below) → nothing


# def remove_combining_with_logging(s):
#     removed = [c for c in s if unicodedata.combining(c) and c != VIRAMA]
#     if removed and VIRAMA not in s:
#         print("⚠️ Removed combining characters (no Virama found):")
#         for c in removed:
#             print(f"  {repr(c)} | U+{ord(c):04X} | {unicodedata.name(c, 'UNKNOWN')} | combining={unicodedata.combining(c)}")
#     return "".join(c for c in s if not (unicodedata.combining(c) and c != VIRAMA))


# def normalize_text(s):
#     s = unicodedata.normalize("NFKC", s)
#     s = _INVIS.sub("", s)
#     s = _DASH.sub("-", s)       
#     s = _BULLET.sub(".", s)     
#     s = _MULTIDOT.sub("..", s)
#     s = s.replace("|", "।")
#     s = s.replace("||", "॥")
#     s = s.replace("(", "").replace(")", "")
#     s = s.replace('\"', "।")
#     s = _WHITESPACE.sub(" ", s).strip()
#     s = _QUOTES.sub('"', s)
#     s = _STRIP_SINGLE.sub("", s)
#     # s = "".join(c for c in s if not unicodedata.combining(c)) 
#     s = remove_combining_with_logging(s) # removes all combining characters except for VIRAMA
#     # s = s.replace("¯", "") # remove upper dash
#     s = s.replace(" ", "").replace("\u00A0", "").replace("\t", "") # remove all spaces
#     s = _NUKTA.sub("", s) # remove NUKTA
#     # s = s.replace("॥", "..")
#     return s

# def main():
#     with open("data/oldNepali/processed/raw_labels/labels_test_raw.json", encoding="utf8") as f:
#         labels = json.load(f)

#     cleaned = []
#     for rec in labels:
#         orig = rec["text"]
#         norm = normalize_text(orig)
#         rec["text"] = norm        
#         cleaned.append(rec)

#     out_path = "data/oldNepali/processed/processed_labels/labels_test_v2.json"
#     with open(out_path, "w", encoding="utf8") as f:
#         json.dump(cleaned, f, ensure_ascii=False, indent=2)
#     print(f"Wrote cleaned labels to: {out_path}\n")

#     print(f"{'ORIGINAL':40s} ↦ NORMALIZED")
#     print("-"*80)
#     for rec in labels[:5]:
#         before = rec["text"] 
#         after  = normalize_text(before)
#         print(f"{before!r:40s} ↦ {after!r}")

# if __name__ == "__main__":
#     main()