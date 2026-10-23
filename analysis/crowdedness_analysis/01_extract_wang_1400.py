"""
01_extract_wang_1400.py
=======================
Extract the Wang 1400 validation set: Wang (2020)'s ~1600 characters minus the
200 used in Huang (2021), kept only if the (character, word) token is in the
training corpus. Adds a Log_Latency column (raw Latency_Correct is kept as-is).

Output: data/processed/wang_1400_characters.csv
        cols: Item, Character, Word, TOP, Latency_Correct, Log_Latency,
              nOccurrence, nTOP
"""

import pandas as pd
import numpy as np
import os
from pyprojroot.here import here

# ── Paths ────────────────────────────────────────────────────────────────────
WANG_PATH   = here("data/raw/wang2020/Database.xlsx")
HUANG_PATH  = here("data/raw/huang2021/huang_tip_of_pen_character-id.xlsx")
TOKENS_PATH = here("data/processed/tokens_CLD_clean.csv")
OUTPUT_CSV  = here("analysis/crowdedness_analysis/output/wang_1400_characters.csv")
os.makedirs(os.path.dirname(str(OUTPUT_CSV)), exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
df_wang   = pd.read_excel(WANG_PATH, sheet_name="Data")
df_huang  = pd.read_excel(HUANG_PATH)
df_tokens = pd.read_csv(TOKENS_PATH)

huang_chars = set(df_huang["Character"].values)
print(f"Wang total: {len(df_wang)}, Huang chars: {len(huang_chars)}")
assert len(huang_chars) == 200
assert huang_chars.issubset(set(df_wang["Character"]))

# ── 2. Valid (char, word) tokens from training corpus ────────────────────────
token_keys = set(zip(df_tokens["char"], df_tokens["word"]))
print(f"Tokens CLD: {len(token_keys)} unique (char, word) pairs")

# ── 3. Separate Huang vs remaining, filter by corpus ─────────────────────────
df_wang = df_wang.copy()
df_wang["group"] = "wang_1400"
df_wang.loc[df_wang["Character"].isin(huang_chars), "group"] = "huang"

df_remaining = df_wang[df_wang["group"] == "wang_1400"].copy()
df_remaining["in_corpus"] = df_remaining.apply(
    lambda row: (row["Character"], row["Word"]) in token_keys, axis=1)

n_in  = df_remaining["in_corpus"].sum()
n_out = (~df_remaining["in_corpus"]).sum()
print(f"\nRemaining (Wang - Huang): total {len(df_remaining)}, "
      f"in corpus {n_in}, excluded {n_out}")

excluded_chars = set(df_remaining[~df_remaining["in_corpus"]]["Character"].values)
df_wang.loc[df_wang["Character"].isin(excluded_chars), "group"] = "excluded"

counts = df_wang["group"].value_counts()
print(f"\nGroup counts:\n{counts}")
assert counts["huang"] == 200
n_wang1400 = counts["wang_1400"]

# ── 4. Build output with Log_Latency ─────────────────────────────────────────
df_out = (
    df_wang[df_wang["group"] == "wang_1400"]
    [["Item", "Character", "Word", "TOP", "Latency_Correct", "nOccurrence", "nTOP"]]
    .sort_values("TOP", ascending=False)
    .reset_index(drop=True)
)

# Keep raw Latency_Correct; add Log_Latency right after it (consistent w/ prior scripts)
col_idx = df_out.columns.get_loc("Latency_Correct") + 1
df_out.insert(col_idx, "Log_Latency", np.log(df_out["Latency_Correct"]))

n_bad = df_out["Log_Latency"].isna().sum()
if n_bad:
    print(f"  NOTE: {n_bad} tokens have missing/invalid latency (Log_Latency = NaN)")

df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n-> Saved {OUTPUT_CSV} ({len(df_out)} characters)")

# ── 5. Summary ───────────────────────────────────────────────────────────────
for grp in ["huang", "wang_1400"]:
    sub = df_wang[df_wang["group"] == grp]
    lat = sub["Latency_Correct"].dropna()
    print(f"\n  {grp.upper()} (n={len(sub)})")
    print(f"    TOP:      mean={sub['TOP'].mean():.4f}, "
          f"range=[{sub['TOP'].min():.4f}, {sub['TOP'].max():.4f}]")
    print(f"    Latency:  mean={lat.mean():.1f}ms, "
          f"log mean={np.log(lat).mean():.3f}, "
          f"range=[{lat.min():.1f}, {lat.max():.1f}]")

# ---- Completion -----------------------------------------------------------
print("Validation items saved; no figure generated.")
