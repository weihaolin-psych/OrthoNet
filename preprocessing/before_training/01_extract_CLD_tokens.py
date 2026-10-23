"""
01_extract_CLD_tokens.py
========================
Extract character-level tokens from the CLD (Chinese Lexical Database).
For each word, extract: word, frequency, character, pinyin, and position.
Then clean by removing words with characters unrecognizable by ChineseBERT tokenizer.

Input:  CLD raw database (.txt)
Output: tokens_CLD.csv       (all tokens, one row per character-in-word)
        tokens_CLD_clean.csv  (UNK-free subset)

NOTE: This is a reconstructed template based on existing outputs.
      If you have the original script, replace this file with it.
"""

import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURATION — adjust paths here
# ==========================================
# Input
CLD_PATH = "data/raw/CLD_chineselexicaldatabase/chineselexicaldatabase2.1.txt"

# Output
TOKENS_OUTPUT    = "data/processed/tokens_CLD.csv"
TOKENS_CLEAN_OUTPUT = "data/processed/tokens_CLD_clean.csv"

# ChineseBERT model (for UNK filtering)
BERT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# ==========================================
# 1. Load CLD and extract character-level tokens
# ==========================================
print("Step 1: Loading CLD database...")
cols_needed = [
    "Word", "Frequency",
    "C1", "C2", "C3", "C4",
    "C1Pinyin", "C2Pinyin", "C3Pinyin", "C4Pinyin",
]
df_cld = pd.read_csv(CLD_PATH, usecols=cols_needed, on_bad_lines="skip", dtype=str)
df_cld["Frequency"] = pd.to_numeric(df_cld["Frequency"], errors="coerce")
print(f"  Loaded {len(df_cld)} words from CLD.")

# ==========================================
# 2. Flatten to character-level rows
# ==========================================
print("Step 2: Extracting character-level tokens...")
rows = []
for _, row in tqdm(df_cld.iterrows(), total=len(df_cld), desc="Extracting"):
    word = row["Word"]
    freq = row["Frequency"]
    for i in range(1, 5):
        char = row.get(f"C{i}")
        pinyin = row.get(f"C{i}Pinyin")
        if pd.notna(char) and str(char).strip():
            rows.append({
                "word": word,
                "frequency": freq,
                "char": char,
                "pinyin": pinyin if pd.notna(pinyin) else "",
                "position": i - 1,
            })

df_tokens = pd.DataFrame(rows)
df_tokens.to_csv(TOKENS_OUTPUT, index=False, encoding="utf-8-sig")
n_words_all = df_tokens["word"].nunique()
print(f"  Saved {len(df_tokens)} tokens ({n_words_all} unique words) → {TOKENS_OUTPUT}")

# ==========================================
# 3. Filter out words with UNK tokens
# ==========================================
print("Step 3: Filtering words with unrecognizable characters...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def word_has_unk(word_str):
    """Check if any token in the word is [UNK]."""
    token_ids = tokenizer.encode(word_str, add_special_tokens=False)
    return tokenizer.unk_token_id in token_ids

# Check each unique word
unique_words = df_tokens["word"].unique()
unk_words = set()
for w in tqdm(unique_words, desc="Checking UNK"):
    if word_has_unk(str(w)):
        unk_words.add(w)

# Filter
df_clean = df_tokens[~df_tokens["word"].isin(unk_words)].copy()
df_clean.to_csv(TOKENS_CLEAN_OUTPUT, index=False, encoding="utf-8-sig")
n_words_clean = df_clean["word"].nunique()
print(f"  Removed {len(unk_words)} words with UNK tokens.")
print(f"  Saved {len(df_clean)} tokens ({n_words_clean} unique words) → {TOKENS_CLEAN_OUTPUT}")
print("✅ Done!")
