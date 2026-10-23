"""
Step 2a: Extract Entropy for Wang 1400 & Compute Model ΔAcc (10 Random Seeds)
=============================================================================

Purpose:
    For each seed × hidden_dim × epoch combination:
    1. Extract model entropy for Wang 1400 characters
    2. Compute per-epoch accuracy (all characters) and ΔAcc per hidden_dim

Inputs:
    - wang_1400_characters.csv
    - evaluation_CLD_twolayers_hidden{N}_seed{S}.csv

Outputs:
    - wang_1400_entropy_all.csv
      Columns: Character, Word, TOP, Latency_Correct, Seed, hidden_dim, Epoch, Entropy
    - delta_acc_all.csv
      Columns: Seed, hidden_dim, acc_first, acc_last, delta_acc
"""

import pandas as pd
import numpy as np
import os
import glob
import time
import re

# ============================================================
# Configuration: paths and all ten seeds
# ============================================================
WANG1400_CSV = "analysis/sensitivity_twolayers_random10/output/wang_1400_characters.csv"
EVAL_DIR = "models/twolayers_10-random-seeds/checkpoint"
SEEDS = [7, 42, 88, 123, 200, 314, 456, 555, 789, 999]
EXPECTED_HIDDEN_DIMS = [4, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 32, 64, 128, 256]
OUTPUT_DIR = "analysis/sensitivity_twolayers_random10/output"
OUTPUT_ENTROPY_CSV = os.path.join(OUTPUT_DIR, "wang_1400_entropy_all.csv")
OUTPUT_ACC_CSV = os.path.join(OUTPUT_DIR, "delta_acc_all.csv")

# Require the full ten-seed grid before replacing any derived data.
required_evaluations = [os.path.join(EVAL_DIR,
    f"evaluation_CLD_twolayers_hidden{dimension}_seed{seed}.csv")
    for seed in SEEDS for dimension in EXPECTED_HIDDEN_DIMS]
missing_evaluations = [path for path in required_evaluations if not os.path.isfile(path)]
if missing_evaluations:
    raise FileNotFoundError("Missing evaluation CSVs; restore or regenerate the complete grid:\n" + "\n".join(missing_evaluations))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load Wang 1400 character list
# ============================================================
print("Step 1: Loading Wang 1400 characters...")
df_chars = pd.read_csv(WANG1400_CSV)
print(f"  {len(df_chars)} characters loaded")

char_keys = set(zip(df_chars["Character"], df_chars["Word"]))
df_lookup = df_chars[["Character", "Word", "TOP", "Latency_Correct"]].copy()

# ============================================================
# 2. Loop through seeds and evaluation files
# ============================================================
print(f"\nStep 2: Extracting entropy & accuracy across {len(SEEDS)} seeds...")

all_entropy_chunks = []
all_acc_records = []
missing_report = {}

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    pattern = os.path.join(EVAL_DIR, f"evaluation_CLD_twolayers_hidden*_seed{seed}.csv")
    eval_files = sorted(glob.glob(pattern))
    print(f"  Found {len(eval_files)} evaluation files")

    if len(eval_files) == 0:
        raise FileNotFoundError(f"No evaluation inputs for seed {seed}; restore or regenerate the ten-seed grid first.")

    for file_path in eval_files:
        m = re.search(r"hidden(\d+)", os.path.basename(file_path))
        hidden_dim = int(m.group(1))
        t0 = time.time()

        # Read columns needed for both entropy extraction and accuracy
        df_eval = pd.read_csv(
            file_path,
            usecols=["Target Char", "Context Word", "Epoch", "Entropy", "Is Correct"]
        )
        df_eval = df_eval.rename(columns={
            "Target Char": "Character",
            "Context Word": "Word",
        })

        # --- Task 1: Extract entropy for Wang 1400 ---
        df_filtered = df_eval[["Character", "Word", "Epoch", "Entropy"]].merge(
            df_lookup, on=["Character", "Word"], how="inner"
        )

        matched_keys = set(zip(df_filtered["Character"], df_filtered["Word"]))
        missing_keys = char_keys - matched_keys
        if missing_keys:
            missing_report[(seed, hidden_dim)] = list(missing_keys)

        df_filtered["Seed"] = seed
        df_filtered["hidden_dim"] = hidden_dim
        all_entropy_chunks.append(df_filtered)

        # --- Task 2: Compute accuracy per epoch (all characters) ---
        acc_by_epoch = df_eval.groupby("Epoch")["Is Correct"].mean().reset_index()
        acc_by_epoch.columns = ["Epoch", "Accuracy"]
        acc_sorted = acc_by_epoch.sort_values("Epoch")

        all_acc_records.append({
            "Seed": seed,
            "hidden_dim": hidden_dim,
            "acc_first": acc_sorted["Accuracy"].iloc[0],
            "acc_last": acc_sorted["Accuracy"].iloc[-1],
            "delta_acc": acc_sorted["Accuracy"].iloc[-1] - acc_sorted["Accuracy"].iloc[0],
        })

        elapsed = time.time() - t0
        n_chars_matched = df_filtered[["Character", "Word"]].drop_duplicates().shape[0]
        n_epochs = df_filtered["Epoch"].nunique()
        print(f"  seed={seed}, hidden_dim={hidden_dim:>4d}: {n_chars_matched} chars × {n_epochs} epochs "
              f"= {len(df_filtered)} rows, "
              f"ΔAcc={all_acc_records[-1]['delta_acc']:.4f}  ({elapsed:.1f}s)"
              f"{'  ⚠ ' + str(len(missing_keys)) + ' missing' if missing_keys else ''}")

# ============================================================
# 3. Report missing characters
# ============================================================
if missing_report:
    print(f"\n⚠ Missing characters:")
    for (seed, h), keys in missing_report.items():
        print(f"  seed={seed}, hidden_dim={h}: {len(keys)} missing")
        for char, word in sorted(keys)[:5]:
            print(f"    {char} ({word})")
        if len(keys) > 5:
            print(f"    ... and {len(keys) - 5} more")
else:
    print(f"\n✅ All characters matched in every evaluation file.")

# ============================================================
# 4. Save entropy CSV
# ============================================================
print(f"\nStep 3: Saving entropy CSV...")

df_entropy = pd.concat(all_entropy_chunks, ignore_index=True)
df_entropy = df_entropy[["Character", "Word", "TOP", "Latency_Correct", "Seed", "hidden_dim", "Epoch", "Entropy"]]
df_entropy = df_entropy.sort_values(["Seed", "hidden_dim", "Epoch", "Character"]).reset_index(drop=True)

df_entropy.to_csv(OUTPUT_ENTROPY_CSV, index=False, encoding="utf-8-sig")

print(f"  → {OUTPUT_ENTROPY_CSV}")
print(f"    Total rows:         {len(df_entropy)}")
print(f"    Characters:         {df_entropy[['Character','Word']].drop_duplicates().shape[0]}")
print(f"    Seeds:              {sorted(df_entropy['Seed'].unique())}")
print(f"    Hidden dimensions:  {sorted(df_entropy['hidden_dim'].unique())}")
print(f"    File size:          {os.path.getsize(OUTPUT_ENTROPY_CSV)/1024/1024:.1f} MB")

# ============================================================
# 5. Save ΔAcc CSV
# ============================================================
print(f"\nStep 4: Saving ΔAcc CSV...")

df_acc = pd.DataFrame(all_acc_records)
df_acc = df_acc.sort_values(["Seed", "hidden_dim"]).reset_index(drop=True)

df_acc.to_csv(OUTPUT_ACC_CSV, index=False)

print(f"  → {OUTPUT_ACC_CSV}")
print(f"    {len(df_acc)} rows ({df_acc['Seed'].nunique()} seeds × {df_acc['hidden_dim'].nunique()} dims)")

# Print ΔAcc summary
for seed in SEEDS:
    sub = df_acc[df_acc["Seed"] == seed]
    excluded = sub[sub["delta_acc"] < 0.1]["hidden_dim"].tolist()
    print(f"    Seed {seed}: excluded dims = {excluded or 'none'}")

print(f"\n✅ Done!")
