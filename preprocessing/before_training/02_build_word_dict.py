"""
02_build_word_dict.py
=====================
Build a word-centric dictionary from the CLD database.
Uses the cleaned token list (from 01_extract_CLD_tokens.py) as a whitelist
to include only words whose characters are all recognizable by BERT.

Input:  CLD raw database + tokens_CLD_clean.csv
Output: word_centric_dict.json

Dictionary structure:
{
    "自然": {
        "frequency": 115.3669,
        "characters": [
            {"char": "自", "pinyin": "zi4", "position": 0},
            {"char": "然", "pinyin": "ran2", "position": 1}
        ]
    },
    ...
}
"""

import pandas as pd
import json
import os
from tqdm import tqdm

# ==========================================
# CONFIGURATION — adjust paths here
# ==========================================
# Input
CLD_PATH         = "data/raw/CLD_chineselexicaldatabase/chineselexicaldatabase2.1.txt"
CLEAN_TOKENS_PATH = "data/processed/tokens_CLD_clean.csv"

# Output
OUTPUT_DIR  = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "word_centric_dict.json")

# ==========================================
# 1. Load clean word whitelist
# ==========================================
print("Step 1: Loading clean word whitelist...")
df_clean = pd.read_csv(CLEAN_TOKENS_PATH)
valid_words = set(df_clean["word"].astype(str))
print(f"  Valid words: {len(valid_words)}")

# ==========================================
# 2. Load CLD metadata
# ==========================================
print("Step 2: Loading CLD database...")
cols_needed = [
    "Word", "Frequency",
    "C1", "C2", "C3", "C4",
    "C1Pinyin", "C2Pinyin", "C3Pinyin", "C4Pinyin",
]
df_cld = pd.read_csv(CLD_PATH, usecols=cols_needed, on_bad_lines="skip", dtype=str)
df_cld["Frequency"] = pd.to_numeric(df_cld["Frequency"], errors="coerce")

# Filter to valid words only
df_target = df_cld[df_cld["Word"].isin(valid_words)].copy()
print(f"  Words after filtering: {len(df_target)}")

# ==========================================
# 3. Build word-centric dictionary
# ==========================================
print("Step 3: Building word-centric dictionary...")
word_dict = {}

for _, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Building dict"):
    word = row["Word"]
    freq = row["Frequency"]

    char_list = []
    for i in range(1, 5):
        char_val = row.get(f"C{i}")
        pinyin_val = row.get(f"C{i}Pinyin")
        if pd.notna(char_val) and str(char_val).strip():
            char_list.append({
                "char": char_val,
                "pinyin": pinyin_val if pd.notna(pinyin_val) else "",
                "position": i - 1,  # 0-based index
            })

    if char_list:
        word_dict[word] = {
            "frequency": float(freq) if pd.notna(freq) else 0.0,
            "characters": char_list,
        }

# Sort by frequency (highest first)
word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1]["frequency"], reverse=True))

# ==========================================
# 4. Save
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Step 4: Saving → {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(word_dict, f, ensure_ascii=False, indent=2)

# ==========================================
# 5. Quick sanity check
# ==========================================
print(f"\n--- Sanity Check ---")
print(f"  Total words: {len(word_dict)}")
for i, (word, data) in enumerate(word_dict.items()):
    if i >= 3:
        break
    chars_str = " | ".join(
        f"{c['char']}({c['pinyin']}, pos={c['position']})" for c in data["characters"]
    )
    print(f"  {word} (freq={data['frequency']:.2f}): {chars_str}")

print("✅ Done!")
