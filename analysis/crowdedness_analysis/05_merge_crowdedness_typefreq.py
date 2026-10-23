"""
05_merge_features.py
====================
Merge optimal-lambda crowdedness (from 04) + log type frequency into each
*_trials_with_entropy.csv, writing *_trials_model.csv whose column names match
exactly what tab_crowdedness.R reads. Non-destructive (originals untouched).

After running this, the ONLY change in tab_crowdedness.R is DATA_FILES:
    huang_trials.csv      -> huang_trials_model.csv
    wang_small_trials.csv -> wang_small_trials_model.csv
    wang_full_trials.csv  -> wang_full_trials_model.csv
(nothing else; same data/processed/ folder)

Key rename: 04 writes log_sem_crowdedness_raw / log_sound_crowdedness_raw, but
tab_crowdedness.R's raw panel reads log_sem_crowdedness / log_sound_crowdedness
(no _raw). This script strips that suffix; the _freq columns pass through.
"""

import pandas as pd
import os

# ── Paths (relative to repo root) ────────────────────────────────────────────
CROWDEDNESS_CSV = "analysis/crowdedness_analysis/output/crowdedness_optimal_lambda.csv"  # from 04
TYPE_FREQ_CSV   = "data/processed/type_frequency.csv"

# Output: a NEW file per dataset (originals are never touched).
OUTPUT_DIR    = "data/processed"
OUTPUT_SUFFIX = "_trials_with_entropy_crowdedness.csv"   # e.g. huang_trials_with_entropy_crowdedness.csv

TRIAL_FILES = {
    "huang":      "data/processed/huang_trials_with_entropy.csv",
    "wang_small": "data/processed/wang_small_trials_with_entropy.csv",
    "wang_full":  "data/processed/wang_full_trials_with_entropy.csv",
}

# Rename optimal-csv columns -> the names tab_crowdedness.R expects (strip _raw)
CROWD_RENAME = {
    "log_sem_crowdedness_raw":   "log_sem_crowdedness",
    "log_sound_crowdedness_raw": "log_sound_crowdedness",
    "sem_crowdedness_raw":       "sem_crowdedness",
    "sound_crowdedness_raw":     "sound_crowdedness",
}
# Columns to pull from the optimal csv (carry both value + log, for the record)
CROWD_CARRY = [
    "sem_crowdedness_raw", "log_sem_crowdedness_raw",
    "sound_crowdedness_raw", "log_sound_crowdedness_raw",
    "sem_crowdedness_freq", "log_sem_crowdedness_freq",
    "sound_crowdedness_freq", "log_sound_crowdedness_freq",
]
TYPE_FREQ_COLS = ["type_frequency", "log_type_frequency"]

# Columns tab_crowdedness.R requires to exist in the merged file (sanity check)
TAB_REQUIRED = [
    "log_sem_crowdedness", "log_sound_crowdedness",
    "log_sem_crowdedness_freq", "log_sound_crowdedness_freq",
    "log_type_frequency", "entropy",
    "log_freq", "aoa", "context_familiarity", "phonetic_radical_order",
    "regularity", "n_stroke",
    "top_binary", "top_eligible", "rt_eligible", "rt", "log_rt",
    "subject", "item",
]


def normalize_keys(df):
    return df.rename(columns={"character": "Character", "word": "Word"})


def main():
    print("Merging optimal crowdedness + log type frequency into trial files...\n")

    crowd = normalize_keys(pd.read_csv(CROWDEDNESS_CSV))
    have = [c for c in CROWD_CARRY if c in crowd.columns]
    missing_src = [c for c in CROWD_CARRY if c not in crowd.columns]
    if missing_src:
        print(f"  NOTE: optimal csv has no {missing_src} (carried only: {have})")
    crowd = (crowd[["Character", "Word"] + have]
             .rename(columns=CROWD_RENAME)
             .drop_duplicates(["Character", "Word"]))
    crowd_target = [CROWD_RENAME.get(c, c) for c in have]

    tf = normalize_keys(pd.read_csv(TYPE_FREQ_CSV))[["Character"] + TYPE_FREQ_COLS] \
        .drop_duplicates("Character")

    for name, path in TRIAL_FILES.items():
        if not os.path.exists(path):
            print(f"\n  Skipping {name}: not found ({path})"); continue
        print(f"\nProcessing {name}: {path}")
        df = pd.read_csv(path); n0 = len(df)

        df = df.drop(columns=[c for c in crowd_target + TYPE_FREQ_COLS if c in df.columns],
                     errors="ignore")
        df = df.merge(crowd, on=["Character", "Word"], how="left")
        assert len(df) == n0, f"row count changed (crowd): {n0} -> {len(df)}"
        df = df.merge(tf, on="Character", how="left")
        assert len(df) == n0, f"row count changed (type freq): {n0} -> {len(df)}"

        # match coverage
        n_miss = df["log_sem_crowdedness"].isna().sum() if "log_sem_crowdedness" in df else n0
        print(f"  crowdedness matched: {n0 - n_miss}/{n0} ({100*(n0-n_miss)/n0:.1f}%)")

        # header sanity check against tab_crowdedness.R
        miss = [c for c in TAB_REQUIRED if c not in df.columns]
        if miss:
            print(f"  !!! MISSING columns tab_crowdedness.R needs: {miss}")
        else:
            print("  OK: all columns tab_crowdedness.R reads are present")

        out = os.path.join(OUTPUT_DIR, name + OUTPUT_SUFFIX)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  -> Saved {out}  ({len(df)} rows x {len(df.columns)} cols)")

    print("\nDone. In tab_crowdedness.R, point DATA_FILES at the new "
          "*_trials_with_entropy_crowdedness.csv files.")


if __name__ == "__main__":
    main()
