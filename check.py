
def replace_fake_candrabindu(text):
    """Replace U+0310 (fake candrabindu) with U+0901 (real Devanagari chandrabindu)."""
    return text.replace('\u0310', '\u0901')

text = "छेक्दाहुला̐क्नचल्दा"  # <- has U+0310 in there

fixed = replace_fake_candrabindu(text)
print("Before:", text)
print("After :", fixed)
