"""Average position-specific entropy within seed, then equally across seeds.
Run from repository root: python preprocessing/after_training/average_ablation_entropy.py
Reads existing hidden22 evaluation CSVs; validates all inputs before replacing
three data/ablation CSVs. Old files and change reports are retained in backups/.
No network training or evaluation is performed.
"""
from datetime import datetime
from pathlib import Path
import shutil

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually; nothing was installed.")

# ---- Configuration ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SEEDS = [7, 42, 88, 123, 200, 314, 456, 555, 789, 999]
HIDDEN_DIM = 22
EPOCH = 11
TOLERANCE = 1e-10
KEYS = ["Target Char", "Context Word"]
MODEL_TYPES = {
    "full": ("models/twolayers_10-random-seeds/checkpoint", "twolayers"),
    "meaning": ("models/ablation_10-random-seeds/checkpoint", "meaning"),
    "sound": ("models/ablation_10-random-seeds/checkpoint", "sound"),
}
OUTPUT_DIR = ROOT / "data/ablation"
DATASETS = ["huang", "wang_small", "wang_full"]

# ---- Preserve equal seed weights after averaging repeated positions ----------
def average_one_model(directory, token):
    frames = []
    reference_counts = None
    for seed in SEEDS:
        path = ROOT / directory / f"evaluation_CLD_{token}_hidden{HIDDEN_DIM}_seed{seed}.csv"
        if not path.exists():
            raise ValueError(f"Missing seed file: {path}")
        frame = pd.read_csv(path, usecols=KEYS + ["Epoch", "Entropy"])
        frame = frame.loc[frame["Epoch"] == EPOCH]
        if frame.empty or frame[KEYS].isna().any().any():
            raise ValueError(f"Missing epoch or character/word keys: {path}")
        if not np.isfinite(frame["Entropy"]).all():
            raise ValueError(f"Non-finite entropy: {path}")
        counts = frame.groupby(KEYS).size().sort_index()
        if reference_counts is None:
            reference_counts = counts
        elif not counts.equals(reference_counts):
            raise ValueError(f"Key coverage or position counts differ across seeds: {path}")
        frames.append(frame.groupby(KEYS)["Entropy"].mean().sort_index())
        print(f"{token}: validated seed {seed}", flush=True)
    # Each column is one seed, regardless of the number of positions in a word.
    averaged = pd.concat(frames, axis=1).mean(axis=1).rename("Entropy").reset_index()
    return averaged, reference_counts

# ---- Compare full entropy with canonical main-analysis inputs ----------------
def check_main_entropy(full):
    for dataset in DATASETS:
        path = ROOT / f"data/processed/{dataset}_trials_with_entropy.csv"
        main = pd.read_csv(path, usecols=["Character", "Word", "entropy"])
        main = main.rename(columns={"Character": KEYS[0], "Word": KEYS[1]})
        joined = main.merge(full, on=KEYS, how="left", validate="many_to_one")
        equal = np.isclose(joined["entropy"], joined["Entropy"], atol=TOLERANCE,
                           rtol=0, equal_nan=True)
        if not equal.all():
            examples = joined.loc[~equal].drop_duplicates().head(10)
            raise ValueError(f"Full entropy differs from {path}:\n{examples}")
        print(f"Full entropy matches main analysis: {dataset}")

# ---- Validate old/new changes before any output replacement ------------------
def compare_old(path, new, counts):
    if not path.exists():
        raise ValueError(f"Missing prior entropy file needed for change audit: {path}")
    old = pd.read_csv(path)
    merged = old.merge(new, on=KEYS, how="outer", suffixes=("_old", "_new"),
                       indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Old/new character-word coverage differs: {path}")
    merged = merged.merge(counts.rename("position_count").reset_index(), on=KEYS,
                          validate="one_to_one")
    merged["difference"] = merged["Entropy_new"] - merged["Entropy_old"]
    changed = ~np.isclose(merged["Entropy_old"], merged["Entropy_new"],
                          atol=TOLERANCE, rtol=0)
    if (changed & merged["position_count"].eq(1)).any():
        raise ValueError(f"Unexpected change in a single-position key: {path}")
    return merged.loc[changed].drop(columns="_merge")

# ---- Main: finish validation, back up all old files, then write ----------------
def main():
    results, counts, changes = {}, {}, {}
    paths = {name: OUTPUT_DIR / f"{name}_hidden{HIDDEN_DIM}_ep{EPOCH}.csv"
             for name in MODEL_TYPES}
    for name, (directory, token) in MODEL_TYPES.items():
        results[name], counts[name] = average_one_model(directory, token)
        if not counts[name].equals(counts["full"]):
            raise ValueError(f"Key coverage or position counts differ for {name}")
        changes[name] = compare_old(paths[name], results[name], counts[name])
    check_main_entropy(results["full"])
    backup = OUTPUT_DIR / "backups" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup.mkdir(parents=True)
    for name, path in paths.items():
        shutil.copy2(path, backup / path.name)
        changes[name].to_csv(backup / f"{name}_changes.csv", index=False, encoding="utf-8-sig")
    for name, path in paths.items():
        results[name].to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved {path}; changed character-word keys: {len(changes[name])}")
    print(f"Old inputs and change reports: {backup}")

if __name__ == "__main__":
    main()
