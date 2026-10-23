"""
Step 1: Average Entropy Across 10 Seeds & Merge into Trial Data
===============================================================
For each (hidden_dim, epoch) combination, compute the mean entropy
across 10 random seeds, then merge into Huang and Wang Small trial data.

Input:
    - evaluation_CLD_twolayers_hidden{N}_seed{S}.csv (×10 seeds × 11 dims)
    - data/processed/huang_trials.csv
    - data/processed/wang_small_trials.csv

Output:
    - huang_trials_averaged.csv
    - wang_small_trials_averaged.csv
    Each file = original trial data + averaged entropy columns (entropy_h{dim}_ep{epoch})
"""

import os
import re
import glob
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
EVAL_DIR = "models/twolayers_10-random-seeds/checkpoint"
SEEDS = [7, 42, 88, 123, 200, 314, 456, 555, 789, 999]
HIDDEN_DIMS = [8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 32]
EPOCHS = range(1, 26)

TRIAL_FILES = {
    "huang": "data/processed/huang_trials.csv",
    "wang_small": "data/processed/wang_small_trials.csv",
}

OUTPUT_DIR = "analysis/glmer_sensitivity_twolayers_averaged10seeds/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load entropy from all seeds and average ───────────────────────
def load_and_average_entropy(eval_dir, seeds, hidden_dims, epochs):
    """
    For each (hidden_dim, epoch), load entropy from all seeds,
    compute the mean, and return a wide-format DataFrame.
    """
    seed_frames = {}

    for seed in seeds:
        all_frames = []
        for h in hidden_dims:
            filepath = os.path.join(eval_dir, f"evaluation_CLD_twolayers_hidden{h}_seed{seed}.csv")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Missing evaluation input: {filepath}. Restore or regenerate the complete ten-seed grid first.")

            df = pd.read_csv(filepath, usecols=["Target Char", "Context Word", "Epoch", "Entropy"])
            df = df.rename(columns={"Target Char": "Character", "Context Word": "Word"})
            df = df[df["Epoch"].isin(epochs)].copy()
            df["col_name"] = "entropy_h" + str(h) + "_ep" + df["Epoch"].astype(str)
            all_frames.append(df)

        df_all = pd.concat(all_frames, ignore_index=True)

        # Pivot to wide format
        df_wide = df_all.pivot_table(
            index=["Character", "Word"],
            columns="col_name",
            values="Entropy",
            aggfunc="first"
        ).reset_index()

        seed_frames[seed] = df_wide
        print(f"  Seed {seed}: {len(df_wide)} characters, {len([c for c in df_wide.columns if c.startswith('entropy_')])} entropy columns")

    # Average across seeds
    print(f"\n  Averaging entropy across {len(seeds)} seeds...")
    entropy_cols = [c for c in seed_frames[seeds[0]].columns if c.startswith("entropy_")]

    # Start with the first seed's Character/Word as index
    df_avg = seed_frames[seeds[0]][["Character", "Word"]].copy()

    for col in entropy_cols:
        values = []
        for seed in seeds:
            if col in seed_frames[seed].columns:
                # Align by Character + Word
                merged = df_avg[["Character", "Word"]].merge(
                    seed_frames[seed][["Character", "Word", col]],
                    on=["Character", "Word"],
                    how="left"
                )
                values.append(merged[col].values)
        # Average across seeds
        df_avg[col] = np.nanmean(values, axis=0)

    print(f"  Averaged: {len(df_avg)} characters, {len(entropy_cols)} entropy columns")
    return df_avg


# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Loading and averaging entropy across seeds...")
    df_entropy_avg = load_and_average_entropy(EVAL_DIR, SEEDS, HIDDEN_DIMS, EPOCHS)
    entropy_cols = [c for c in df_entropy_avg.columns if c.startswith("entropy_")]

    for dataset_name, trial_path in TRIAL_FILES.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {dataset_name}")
        print(f"{'='*60}")

        df_trials = pd.read_csv(trial_path)
        print(f"  Original: {len(df_trials):,} rows × {len(df_trials.columns)} cols")

        df_merged = df_trials.merge(
            df_entropy_avg,
            on=["Character", "Word"],
            how="left"
        )

        # Check merge
        n_missing = df_merged[entropy_cols[0]].isna().sum()
        if n_missing > 0:
            print(f"  ⚠ {n_missing} trials have missing entropy")
        else:
            print(f"  ✓ All trials matched")

        print(f"  Merged: {len(df_merged):,} rows × {len(df_merged.columns)} cols")

        # Save
        out_name = f"{dataset_name}_trials_averaged.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df_merged.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  → Saved {out_path} ({size_mb:.1f} MB)")

    print(f"\n✅ Done! {len(TRIAL_FILES)} files saved with averaged entropy.")
