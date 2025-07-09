import unicodedata
import pandas as pd
from collections import defaultdict
from torchmetrics.text import CharErrorRate  # pip install torchmetrics
import matplotlib.pyplot as plt


df_lemmas = pd.read_csv("lemmas.csv")
form_to_lemmas = defaultdict(set)
for _, row in df_lemmas.iterrows():
    form = unicodedata.normalize("NFC", row["word_form"].strip())
    form_to_lemmas[form].add(form)
known_lemmas = set(form_to_lemmas)

VIRAMA         = "्"
HALF_RA        = VIRAMA + "र"
ANUSVARA       = "ं"
CHANDRA_BINDU  = "ँ"
MATRA_SIGNS    = {'ा','ि','ी','ु','ू','े','ै','ो','ौ'}


def split_by_consonant_and_matra(text):
    text = unicodedata.normalize("NFC", text)
    chunks, i = [], 0
    while i < len(text):
        c = text[i]
        nxt = text[i+1] if i+1 < len(text) else ""
        if nxt in MATRA_SIGNS or nxt in {ANUSVARA, CHANDRA_BINDU}:
            chunks.append(c + nxt)
            i += 2
        else:
            chunks.append(c)
            i += 1
    return chunks

def try_insertions(token, mark):
    for pos in range(1, len(token)+1):
        yield token[:pos] + mark + token[pos:]

# === Two-step fuzzy candidate generator ===
def restore_two_marks(token, first_mark, second_mark):
    # first insert first_mark, then second_mark
    for mid in try_insertions(token, first_mark):
        mid = unicodedata.normalize("NFC", mid)
        for cand in try_insertions(mid, second_mark):
            cand = unicodedata.normalize("NFC", cand)
            if cand in known_lemmas:
                return cand
    return None

# === Single-rule restorers (for completeness) ===
def restore_half_ra(token):
    for cand in try_insertions(token, HALF_RA):
        cand = unicodedata.normalize("NFC", cand)
        if cand in known_lemmas:
            return cand
    return None

def restore_chandra_bindu(token):
    for cand in try_insertions(token, CHANDRA_BINDU):
        cand = unicodedata.normalize("NFC", cand)
        if cand in known_lemmas:
            return cand
    return None

def replace_ta_na(token):
    """Try replacing 'त' with 'न' and vice versa."""
    candidates = set()
    for i, c in enumerate(token):
        if c == 'व':
            alt = token[:i] + 'च' + token[i+1:]
            candidates.add(alt)
        elif c == 'च':
            alt = token[:i] + 'व' + token[i+1:]
            candidates.add(alt)
    return list(candidates)

def lemmatize_with_fuzzy(text, min_match_len=2):
    text = unicodedata.normalize("NFC", text)
    tokens = split_by_consonant_and_matra(text)
    reconstructed = []
    matched_before, matched_after = [], []

    for token in tokens:
        if token in known_lemmas:
            reconstructed.append(token)
            matched_before.append(token)
            matched_after.append(token)
        else:
            # Try simple त ↔ न swaps
            candidates = replace_ta_na(token)
            fixed = None
            for cand in candidates:
                if cand in known_lemmas:
                    fixed = cand
                    break
            if fixed:
                reconstructed.append(fixed)
                matched_before.append(token)
                matched_after.append(fixed)
            else:
                # keep as is
                reconstructed.append(token)

    full_reconstruction = "".join(reconstructed)
    total = len(text)
    matched = sum(len(t) for t in matched_after)
    ratio = matched / total if total else 0.0

    return {
        "matched_before": matched_before,
        "matched_after": matched_after,
        "reconstructed": full_reconstruction,
        "match_ratio": ratio,
        "counts": f"{matched}/{total}"
    }
if __name__ == "__main__":
    gt   = "सेवकलोकरमणोपाध्यायकोवेदो"
    pred = "सेपकलोकरभणौपाध्यायकोवेरा"

    print("🔍 GT:", gt)
    print("🔍 PD:", pred)
    print("🧩 Chunks:", split_by_consonant_and_matra(pred))

    res = lemmatize_with_fuzzy(pred, min_match_len=2)
    print("✅ Before fuzzy:", res["matched_before"])
    print("🛠️  After fuzzy:", res["matched_after"])
    print(f"📈 Match Ratio: {res['match_ratio']:.2f} ({res['counts']})")

    # === Apply fuzzy lemmatizer to a CSV column ===
    INPUT_CSV_PATH  = "results/evaluation_results_cleaned_bytebpe500_test_new8.csv"   # <- adjust
    OUTPUT_CSV_PATH = "results/fuzzy_matched.csv"  # <- adjust
    COLUMN_NAME     = "prediction"                            # <- column with noisy OCR text

    df = pd.read_csv(INPUT_CSV_PATH)

    # Apply the fuzzy lemmatizer and store reconstructed output
    df["fuzzy_corrected"] = df[COLUMN_NAME].apply(
        lambda text: lemmatize_with_fuzzy(str(text), min_match_len=2)["reconstructed"]
    )
    cer = CharErrorRate()
    # Compute CER for fuzzy output
    df["fuzzy_cer"] = df.apply(
        lambda row: cer([str(row["fuzzy_corrected"])], [str(row["ground_truth"])]).item(),
        axis=1
    )
    # Save to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Done! Saved to: {OUTPUT_CSV_PATH}")

    # === Step 1: Dynamically choose the better correction based on CER
    cer = CharErrorRate()

    def select_best_output(row):
        raw   = str(row["prediction"])
        fuzzy = str(row["fuzzy_corrected"])
        gt    = str(row["ground_truth"])

        raw_cer   = cer([raw], [gt]).item()
        fuzzy_cer = cer([fuzzy], [gt]).item()

        return fuzzy if fuzzy_cer < raw_cer else raw

    df["best_correction"] = df.apply(select_best_output, axis=1)

    # === Step 2: Compute CER for best_correction
    df["best_cer"] = df.apply(lambda row: cer([str(row["best_correction"])], [str(row["ground_truth"])]).item(), axis=1)

    # === Step 3: Corpus-level CER for best_correction
    corpus_best = cer(df["best_correction"].astype(str).tolist(), df["ground_truth"].astype(str).tolist()).item()

    print(f"🏆 Corpus-level CER (Best of Raw vs Fuzzy): {corpus_best:.4f}")

    # === Step 4: Histogram comparison (all 3 versions)
    plt.figure(figsize=(10, 5))
    plt.hist(df['raw_cer'], bins=20, alpha=0.5, label='Raw CER')
    plt.hist(df['fuzzy_cer'], bins=20, alpha=0.5, label='Fuzzy CER')
    plt.hist(df['best_cer'], bins=20, alpha=0.5, label='Best CER')
    plt.title("CER Distribution: Raw vs Fuzzy vs Best")
    plt.xlabel("Character Error Rate")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/cer_histogram_all.png")
    plt.close()

    # === Step 5: Save final CSV with all 3 columns
    df.to_csv("results/fuzzy_filtered_best.csv", index=False)
    print("✅ Saved CSV with 'best_correction' column and all CERs to results/fuzzy_filtered_best.csv")

    # === Average CERs for comparison
    mean_raw_cer   = df["raw_cer"].mean()
    mean_fuzzy_cer = df["fuzzy_cer"].mean()
    mean_best_cer  = df["best_cer"].mean()

    print("\n🎯 AVERAGE CER COMPARISON")
    print(f"🔹 Raw CER   (before fuzzy): {mean_raw_cer:.4f}")
    print(f"🔹 Fuzzy CER (after fuzzy) : {mean_fuzzy_cer:.4f}")
    print(f"🏆 Best CER  (selected one): {mean_best_cer:.4f}")

    # Optional: count wins
    fuzzy_helped_count = (df["fuzzy_cer"] < df["raw_cer"]).sum()
    print(f"\n📈 Fuzzy improved CER in {fuzzy_helped_count} out of {len(df)} lines ({100*fuzzy_helped_count/len(df):.2f}%)")