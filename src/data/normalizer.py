import json
import unicodedata
import re

# _DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

VIRAMA = "\u094D" 
_DIGITS      = str.maketrans("०१२३४५६७८९０１２３４５６７８९", "01234567890123456789")
_INVIS       = re.compile(r"[\u200B-\u200D\u00AD]")    
_DASH        = re.compile(r"[–—−]")          
_BULLET      = re.compile(r"[·•‧∙]")         
_MULTIDOT    = re.compile(r"\.{3,}")         
_WHITESPACE  = re.compile(r"[ \t\u00A0]+")   
_QUOTES        = re.compile(r'[“”]')         
_STRIP_SINGLE  = re.compile(r"[`'‘’]")   
_NUKTA = re.compile(r"\u093C") # NUKTA (dot below) → nothing
_DIGIT_RUN    = re.compile(r"[0-9०१२३४५६७८९]+")


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
    s = s.translate(_DIGITS)
    # s = _DIGIT_RUN.sub("<NUM>", s)
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
    with open("data/oldNepali/processed/labels.json", encoding="utf8") as f:
        labels = json.load(f)

    cleaned = []
    for rec in labels:
        orig = rec["text"]
        norm = normalize_text(orig)
        rec["text"] = norm        
        cleaned.append(rec)

    out_path = "data/oldNepali/processed/labels_cleaned.json"
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
