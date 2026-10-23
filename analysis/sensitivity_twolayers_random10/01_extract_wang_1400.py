"""
Step 1: Extract Wang 1400 Validation Set
=========================================

Purpose:
    From Wang et al. (2020)'s 1,600 characters, remove the 200 characters
    used in Huang et al. (2021). Use ALL remaining characters as an
    independent validation set for hyperparameter selection.

Outputs:
    - wang_1400_characters.csv             : Character list
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 0. Paths
# ============================================================
WANG_PATH   = "data/raw/wang2020/Database.xlsx"
HUANG_PATH  = "data/raw/huang2021/huang_tip_of_pen_character-id.xlsx"
TOKENS_PATH = "data/processed/tokens_CLD_clean.csv"
OUTPUT_DIR  = "analysis/sensitivity_twolayers_random10/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
df_wang   = pd.read_excel(WANG_PATH, sheet_name="Data")
df_huang  = pd.read_excel(HUANG_PATH)
df_tokens = pd.read_csv(TOKENS_PATH)

huang_chars = set(df_huang["Character"].values)
print(f"Wang total: {len(df_wang)}, Huang chars: {len(huang_chars)}")
assert len(huang_chars) == 200
assert huang_chars.issubset(set(df_wang["Character"]))

# ============================================================
# 2. Build valid (char, word) set from training corpus
# ============================================================
token_keys = set(zip(df_tokens["char"], df_tokens["word"]))
print(f"Tokens CLD: {len(token_keys)} unique (char, word) pairs")

# ============================================================
# 3. Separate Huang vs remaining, filter by corpus
# ============================================================
df_wang = df_wang.copy()
df_wang["group"] = "wang_1400"
df_wang.loc[df_wang["Character"].isin(huang_chars), "group"] = "huang"

df_remaining = df_wang[df_wang["group"] == "wang_1400"].copy()
df_remaining["in_corpus"] = df_remaining.apply(
    lambda row: (row["Character"], row["Word"]) in token_keys, axis=1
)

n_in  = df_remaining["in_corpus"].sum()
n_out = (~df_remaining["in_corpus"]).sum()
print(f"\nRemaining characters (Wang - Huang):")
print(f"  Total:                  {len(df_remaining)}")
print(f"  In training corpus:     {n_in}")
print(f"  NOT in training corpus: {n_out}")

if n_out > 0:
    excluded = df_remaining[~df_remaining["in_corpus"]][["Character", "Word", "TOP"]]
    excluded = excluded.sort_values("TOP", ascending=False)
    print(f"  Excluded from analysis:")
    for _, row in excluded.iterrows():
        print(f"    {row['Character']} ({row['Word']}) TOP={row['TOP']:.4f}")

excluded_chars = set(df_remaining[~df_remaining["in_corpus"]]["Character"].values)
df_wang.loc[df_wang["Character"].isin(excluded_chars), "group"] = "excluded"

counts = df_wang["group"].value_counts()
print(f"\nFinal group counts:\n{counts}")
assert counts["huang"] == 200
n_wang1400 = counts["wang_1400"]

# ============================================================
# 4. Summary statistics
# ============================================================
print("\n" + "=" * 55)
for grp in ["huang", "wang_1400"]:
    sub = df_wang[df_wang["group"] == grp]
    print(f"\n  {grp.upper()} (n={len(sub)})")
    print(f"  TOP rate:  mean={sub['TOP'].mean():.4f}, median={sub['TOP'].median():.4f}, "
          f"range=[{sub['TOP'].min():.4f}, {sub['TOP'].max():.4f}]")
    lat = sub["Latency_Correct"].dropna()
    print(f"  Latency:   mean={lat.mean():.1f}ms, median={lat.median():.1f}ms, "
          f"range=[{lat.min():.1f}, {lat.max():.1f}]")
print("=" * 55)

# ============================================================
# 5. Save Wang 1400 character list
# ============================================================
df_out = (
    df_wang[df_wang["group"] == "wang_1400"]
    [["Item", "Character", "Word", "TOP", "Latency_Correct", "nOccurrence", "nTOP"]]
    .sort_values("TOP", ascending=False)
    .reset_index(drop=True)
)

out_csv = os.path.join(OUTPUT_DIR, "wang_1400_characters.csv")
df_out.to_csv(out_csv, index=False)
print(f"\n→ Saved {out_csv} ({len(df_out)} characters)")

# ============================================================
print("Validation items saved; no figure generated.")
