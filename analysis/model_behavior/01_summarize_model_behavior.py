"""Describe entropy learning and item distributions from existing evaluations.

Run: python analysis/model_behavior/01_summarize_model_behavior.py
Reads hidden22 evaluations and canonical processed human-data item lists.
Writes descriptive/audit CSVs and run metadata under model_behavior/output.
Does not train, infer, fit human data, or generate manuscript figures/tables.
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually.")

# ---- Configuration ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "models/twolayers_10-random-seeds/checkpoint"
OUTPUT_DIR = ROOT / "analysis/model_behavior/output"
SEEDS = [7, 42, 88, 123, 200, 314, 456, 555, 789, 999]
EPOCHS = list(range(1, 26))
HIDDEN_DIM = 22
SELECTED_EPOCH = 11
ENTROPY_TOLERANCE = 1e-10
DATASETS = {
    "Huang": "huang", "Wang Small": "wang_small", "Wang Full": "wang_full"
}
EXPECTED_COUNTS = {"Huang": (200, 200), "Wang Small": (200, 200),
                   "Wang Full": (1600, 1592)}
KEYS = ["Character", "Word"]
READ_COLUMNS = ["Epoch", "Target Char", "Context Word", "Entropy"]
SD_DDOF = 1
QUANTILE_PROBABILITIES = [0.25, 0.50, 0.75]
QUANTILE_METHOD = "linear"
SD_LEVEL = "items_after_seed_averaging"
METADATA_FILE = "entropy_run_metadata.json"

# ---- Input validation and position-to-item aggregation ---------------------
def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate_evaluation(frame):
    """Average repeated positions within each epoch and Character–Word item."""
    require(not frame.empty, "Empty evaluation file.")
    require(not frame[READ_COLUMNS].isna().any().any(), "Missing evaluation values.")
    require(set(frame["Epoch"]) == set(EPOCHS), "Unexpected or missing epochs.")
    require(np.isfinite(frame["Entropy"]).all() and frame["Entropy"].ge(0).all(),
            "Entropy must be finite and nonnegative.")
    frame = frame.rename(columns={"Target Char": "Character", "Context Word": "Word",
                                  "Entropy": "entropy", "Epoch": "epoch"})
    # Repeated target positions are legitimate observations, not duplicate errors.
    return frame.groupby(["epoch"] + KEYS, sort=True).agg(
        entropy=("entropy", "mean"), position_count=("entropy", "size"))


def load_items(path):
    frame = pd.read_csv(path, usecols=KEYS + ["entropy"])
    require(not frame[KEYS].isna().any().any(), f"Missing item keys: {path}")
    unique = frame.drop_duplicates()
    require(not unique.duplicated(KEYS).any(), f"Inconsistent item entropy: {path}")
    observed = unique["entropy"].dropna()
    require(np.isfinite(observed).all() and observed.ge(0).all(),
            f"Invalid canonical entropy: {path}")
    return unique.set_index(KEYS).sort_index()


def describe_items(values):
    """Describe the distribution of item entropies already averaged across seeds."""
    require(len(values) > SD_DDOF and np.isfinite(values).all(),
            "Insufficient or invalid items for a sample SD.")
    q25, median, q75 = np.quantile(values, QUANTILE_PROBABILITIES, method=QUANTILE_METHOD)
    return dict(n_items=len(values), n_seeds=len(SEEDS), entropy_mean=values.mean(),
                entropy_item_sd=values.std(ddof=SD_DDOF), entropy_q25=q25,
                entropy_median=median, entropy_q75=q75)


def check_previous_means(summary, filename, keys):
    """Optional audit against existing seed-level summaries; never use their SD."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        return "not available"
    previous = pd.read_csv(path)
    compared = summary.merge(previous[keys + ["entropy_mean"]], on=keys,
                             how="outer", suffixes=("_new", "_old"),
                             indicator=True, validate="one_to_one")
    require(compared["_merge"].eq("both").all() and np.isclose(
        compared.entropy_mean_new, compared.entropy_mean_old,
        atol=ENTROPY_TOLERANCE, rtol=0).all(),
        f"Mean entropy differs from historical output: {path}")
    return "means match within absolute tolerance; historical SD not used"

# ---- Extract all epochs once per seed; retain selected-epoch item outputs ---
def main():
    require(SELECTED_EPOCH in EPOCHS and len(SEEDS) == len(set(SEEDS)) >= 2,
            "Invalid epoch or seed configuration.")
    evaluation_paths = {
        seed: EVALUATION_DIR / f"evaluation_CLD_twolayers_hidden{HIDDEN_DIM}_seed{seed}.csv"
        for seed in SEEDS
    }
    dataset_paths = {name: ROOT / f"data/processed/{stem}_trials_with_entropy.csv"
                     for name, stem in DATASETS.items()}
    for path in [*evaluation_paths.values(), *dataset_paths.values()]:
        require(path.is_file(), f"Missing input: {path}")
    items = {name: load_items(path) for name, path in dataset_paths.items()}
    # Accumulate one aligned entropy vector per epoch, not all seed dataframes.
    entropy_sums = {}
    reference_counts = None
    for seed, path in evaluation_paths.items():
        print(f"Reading seed {seed}: {path.name}", flush=True)
        frame = pd.read_csv(path, usecols=READ_COLUMNS)
        grouped = aggregate_evaluation(frame)
        del frame
        for epoch in EPOCHS:
            epoch_items = grouped.xs(epoch, level="epoch")
            counts = epoch_items.position_count
            if reference_counts is None:
                reference_counts = counts
            require(counts.equals(reference_counts),
                    f"Item coverage/position counts differ: seed {seed}, epoch {epoch}.")
            if epoch not in entropy_sums:
                entropy_sums[epoch] = epoch_items.entropy.to_numpy(copy=True)
            else:
                entropy_sums[epoch] += epoch_items.entropy.to_numpy()
        del grouped

    # Average seeds for each item before computing the between-item sample SD.
    learning_records = []
    for epoch in EPOCHS:
        entropy_sums[epoch] /= len(SEEDS)
        learning_records.append(dict(epoch=epoch, **describe_items(entropy_sums[epoch])))
    selected_items = reference_counts.to_frame()
    selected_items["entropy"] = entropy_sums[SELECTED_EPOCH]
    selected_items["epoch"] = SELECTED_EPOCH
    selected_items["n_seeds"] = len(SEEDS)

    # The red subset must have exactly the same keys and values in both datasets.
    require(items["Huang"].equals(items["Wang Small"]),
            "Huang and Wang Small no longer have identical item entropy.")
    shared_keys = items["Huang"].index
    require(shared_keys.isin(items["Wang Full"].index).all(),
            "Shared items are not all present in Wang Full.")
    coverage_frames, dataset_records = [], []
    for dataset, canonical in items.items():
        comparison = canonical.rename(columns={"entropy": "canonical_entropy"}).join(
            selected_items[["entropy"]])
        comparison["available"] = comparison.entropy.notna()
        require(comparison.available.equals(canonical.entropy.notna()),
                f"Model/canonical availability differs: {dataset}.")
        require((len(canonical), int(comparison.available.sum())) == EXPECTED_COUNTS[dataset],
                f"Unexpected item coverage: {dataset}.")
        require(np.isclose(comparison.canonical_entropy, comparison.entropy,
                           atol=ENTROPY_TOLERANCE, rtol=0, equal_nan=True).all(),
                f"Selected-epoch mean entropy differs from main analysis: {dataset}.")
        comparison["entropy_difference"] = comparison.entropy - comparison.canonical_entropy
        comparison["shared_item"] = comparison.index.isin(shared_keys)
        dataset_records.append(dict(dataset=dataset, epoch=SELECTED_EPOCH,
                                    n_total=len(canonical),
                                    n_missing=int((~comparison.available).sum()),
                                    **describe_items(comparison.loc[comparison.available, "entropy"])))
        coverage_frames.append(comparison.reset_index().assign(dataset=dataset))
    learning = pd.DataFrame(learning_records)
    dataset_summary = pd.DataFrame(dataset_records)
    audit = pd.concat(coverage_frames, ignore_index=True)
    ranked = audit.loc[audit.dataset.eq("Wang Full") & audit.available].sort_values(
        ["entropy"] + KEYS, ascending=[False, True, True]).copy()
    ranked["wang_full_rank"] = np.arange(1, len(ranked) + 1)
    audit = audit.merge(ranked[KEYS + ["wang_full_rank"]], on=KEYS, how="left",
                        validate="many_to_one")
    audit["wang_full_rank"] = audit.wang_full_rank.astype("Int64")
    previous_checks = {
        "learning": check_previous_means(learning, "learning_summary.csv", ["epoch"]),
        "datasets": check_previous_means(dataset_summary, "dataset_summary.csv", ["dataset", "epoch"]),
    }
    outputs = {
        "learning_entropy_summary.csv": learning,
        "selected_epoch_item_entropy.csv": selected_items.reset_index(),
        "dataset_entropy_summary.csv": dataset_summary,
        "dataset_entropy_item_audit.csv": audit,
    }

    # Write only after all scientific checks pass; metadata marks a complete run.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / METADATA_FILE
    if metadata_path.exists():
        metadata_path.unlink()
    for filename, values in outputs.items():
        values.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
        print(f"Saved {filename}: {len(values)} rows", flush=True)
    metadata = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), hidden_dim=HIDDEN_DIM,
        selected_epoch=SELECTED_EPOCH, epochs=EPOCHS, seeds=SEEDS,
        entropy_tolerance=ENTROPY_TOLERANCE, sd_ddof=SD_DDOF, sd_level=SD_LEVEL,
        quantile_probabilities=QUANTILE_PROBABILITIES, quantile_method=QUANTILE_METHOD,
        item_unit="Character-Word; average repeated positions within seed first",
        weighting="equal items within seed, equal seeds; no frequency/trial weighting",
        entropy_unit="nats", datasets=list(DATASETS), historical_mean_checks=previous_checks,
        expected_counts=EXPECTED_COUNTS,
        summary_sha256={name: hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
                        for name in outputs},
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
        inputs=[dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                     modified_ns=path.stat().st_mtime_ns)
                for path in [*evaluation_paths.values(), *dataset_paths.values()]],
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Completed. Run fig_model_behavior.py; item-level M/SD are in dataset_entropy_summary.csv.")


if __name__ == "__main__":
    main()
