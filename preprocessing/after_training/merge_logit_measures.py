"""Merge three context-matched, seed-standardized logit measures into trials.

Run: python -X utf8 preprocessing/after_training/merge_logit_measures.py
Reads data/logit_measures/*.csv and canonical *_trials_with_entropy.csv files.
Writes three *_trials_with_logit_measures.csv files plus preparation audits.
No training, inference, behavioral fitting or plotting. Requires NumPy/pandas;
missing dependencies are reported without automatic installation.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually.")

# ---- Configuration ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data/logit_measures"
DATA_DIR = ROOT / "data/processed"
AUDIT_DIR = ROOT / "data/analysis_inputs/relative_measure_preparation"
DATASETS = ("huang", "wang_small", "wang_full")
SEEDS = (7, 42, 88, 123, 200, 314, 456, 555, 789, 999)
HIDDEN_DIM = 22
EPOCH = 11
EXPECTED_POSITIONS = 106_844
EXPECTED_CONTEXTS = 105_617
EXPECTED_CHARACTERS = 4_677
EXPECTED_ITEMS = {"huang": (200, 200), "wang_small": (200, 200), "wang_full": (1600, 1592)}
KEYS = ["Character", "Word"]
METRICS = ["target_logit_measure", "deficit_measure", "blocking_measure"]
STANDARD_DEVIATION_DDOF = 1
ATOL = 1e-10
RTOL = 1e-12
CSV_ENCODING = "utf-8-sig"
CSV_FLOAT_FORMAT = "%.17g"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- Validate and aggregate the model data independently of plotting --------
def z_score(values, description):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), f"Nonfinite values: {description}")
    mean = values.mean()
    sd = values.std(ddof=STANDARD_DEVIATION_DDOF)
    require(np.isfinite(sd) and sd > 0, f"Zero/invalid SD: {description}")
    standardized = (values - mean) / sd
    require(abs(standardized.mean()) < ATOL and
            abs(standardized.std(ddof=STANDARD_DEVIATION_DDOF) - 1) < ATOL,
            f"Z-score validation failed: {description}")
    return standardized, float(mean), float(sd)


def build_item_measures(summary, positions):
    require(set([*KEYS, "deficit_measure", "blocking_measure", "n_positions", "n_seeds"]).issubset(summary),
            "Missing Character-Word summary columns.")
    require(len(summary) == EXPECTED_CONTEXTS and summary.Character.nunique() == EXPECTED_CHARACTERS and
            not summary.duplicated(KEYS).any() and summary[KEYS].ne("").all().all(),
            "Invalid summary inventory/keys.")
    require(summary.n_seeds.eq(len(SEEDS)).all() and summary.n_positions.ge(1).all() and
            summary.n_positions.mod(1).eq(0).all() and summary.n_positions.sum() == EXPECTED_POSITIONS,
            "Invalid summary seed/position coverage.")
    require(np.isfinite(summary[["deficit_measure", "blocking_measure"]].to_numpy()).all(),
            "Nonfinite prepared relative measures.")
    needed = ["seed", *KEYS, "Position", "target_id", "target_logit", "deficit_z", "blocking_z"]
    require(set(needed).issubset(positions), "Missing position-level columns.")
    require(len(positions) == EXPECTED_POSITIONS * len(SEEDS) and
            set(positions.seed.unique()) == set(SEEDS), "Incomplete position/seed inventory.")
    require(not positions.duplicated(["seed", "Word", "Position"]).any() and
            positions[KEYS].ne("").all().all(), "Invalid original position keys.")
    for column in ("Position", "target_id"):
        require(np.isfinite(positions[column]).all() and positions[column].ge(0).all() and
                positions[column].mod(1).eq(0).all(), f"Invalid {column} indices.")
    require(positions.target_id.lt(EXPECTED_CHARACTERS).all(), "Target column out of bounds.")
    require(np.isfinite(positions[["target_logit", "deficit_z", "blocking_z"]].to_numpy()).all(),
            "Nonfinite position-level measures.")
    mapping = positions[["Character", "target_id"]].drop_duplicates()
    require(len(mapping) == EXPECTED_CHARACTERS and not mapping.Character.duplicated().any() and
            not mapping.target_id.duplicated().any(), "Inconsistent character/target-column mapping.")

    reference = None
    per_seed, normalization = [], []
    for seed in SEEDS:
        part = positions.loc[positions.seed.eq(seed)].copy()
        identity = part[[*KEYS, "Position", "target_id"]].reset_index(drop=True)
        require(len(part) == EXPECTED_POSITIONS, f"Incomplete positions: seed {seed}")
        if reference is None:
            reference = identity
        else:
            pd.testing.assert_frame_equal(identity, reference, check_exact=True)
        # Normalize raw target logits BEFORE repeated-position or seed averaging.
        part["target_logit_measure"], mean, sd = z_score(part.target_logit.to_numpy(), f"target seed {seed}")
        normalization.append(dict(seed=seed, n_positions=len(part), target_mean=mean, target_sd=sd))
        for column in ("deficit_z", "blocking_z"):
            require(abs(part[column].mean()) < ATOL and
                    abs(part[column].std(ddof=STANDARD_DEVIATION_DDOF) - 1) < ATOL,
                    f"Prepared {column} is not seed-standardized: seed {seed}")
        values = part.groupby(KEYS, sort=True).agg(
            target_logit_measure=("target_logit_measure", "mean"),
            deficit_measure=("deficit_z", "mean"), blocking_measure=("blocking_z", "mean"),
            n_positions=("Position", "size"),
        ).reset_index()
        require(len(values) == EXPECTED_CONTEXTS, f"Context inventory differs: seed {seed}")
        values["seed"] = seed
        per_seed.append(values)
    grouped = pd.concat(per_seed, ignore_index=True).groupby(KEYS, sort=True)
    require(grouped.n_positions.nunique().eq(1).all(), "Position multiplicities differ across seeds.")
    recomputed = grouped[METRICS].mean()
    recomputed["n_positions"] = grouped.n_positions.first()
    recomputed["n_seeds"] = grouped.seed.nunique()
    source = summary.set_index(KEYS).sort_index()
    require(recomputed.index.equals(source.index), "Model summary/position keys differ.")
    for name in ("n_positions", "n_seeds"):
        require(np.array_equal(recomputed[name], source[name]), f"Model coverage differs: {name}")
    require(np.allclose(recomputed[["deficit_measure", "blocking_measure"]],
                        source[["deficit_measure", "blocking_measure"]], atol=ATOL, rtol=RTOL),
            "Prepared relative measures cannot be reproduced from position CSV.")
    # Keep the existing summary values, rather than replacing their last bits.
    result = source[["deficit_measure", "blocking_measure", "n_positions", "n_seeds"]].copy()
    result["target_logit_measure"] = recomputed.target_logit_measure
    return result.reset_index()[[*KEYS, *METRICS, "n_positions", "n_seeds"]], normalization


# ---- Preserve all original trial fields and order during the left join -------
def merge_trials(dataset, source, measures):
    require(not set(METRICS).intersection(source.columns), "Source already contains new measures.")
    require(set([*KEYS, "entropy"]).issubset(source), "Missing canonical trial columns.")
    require(source[KEYS].notna().all().all() and source[KEYS].ne("").all().all(), "Missing trial keys.")
    require(not measures.duplicated(KEYS).any(), "Duplicate model merge keys.")
    original_columns = source.columns.tolist()
    result = source.merge(measures[[*KEYS, *METRICS]], on=KEYS,
                          how="left", sort=False, validate="many_to_one")
    pd.testing.assert_frame_equal(source.reset_index(drop=True),
                                  result[original_columns].reset_index(drop=True), check_exact=True)
    present = result[METRICS].notna()
    require((present.all(axis=1) | ~present.any(axis=1)).all(), "Partially missing new measures.")
    used = present.all(axis=1)
    require(np.isfinite(result.loc[used, METRICS].to_numpy()).all(), "Nonfinite merged measures.")
    entropy = pd.to_numeric(source.entropy.replace("", np.nan), errors="raise")
    require(np.array_equal(entropy.notna().to_numpy(), used.to_numpy()),
            f"Measure context coverage differs from canonical entropy: {dataset}")
    inventory = result[[*KEYS, *METRICS]].drop_duplicates()
    require(not inventory.duplicated(KEYS).any(), "Within-item predictor values differ.")
    total, matched = EXPECTED_ITEMS[dataset]
    observed = int(inventory[METRICS[0]].notna().sum())
    require(len(inventory) == total and observed == matched, f"Unexpected item coverage: {dataset}")
    coverage = dict(dataset=dataset, n_trials=len(source), n_items=len(inventory),
                    n_matched_items=observed, n_missing_items=total - observed,
                    n_missing_trials=int((~used).sum()))
    absent = inventory.loc[inventory[METRICS[0]].isna(), KEYS].copy()
    absent.insert(0, "dataset", dataset)
    return result[original_columns + METRICS], coverage, absent


# ---- Validate everything before publishing trial files and provenance --------
def main():
    require(len(set(SEEDS)) == len(SEEDS), "Duplicate seeds in Configuration.")
    summary_path = MODEL_DIR / "character_word_measures.csv"
    position_path = MODEL_DIR / "position_metrics_by_seed.csv"
    trial_paths = {dataset: DATA_DIR / f"{dataset}_trials_with_entropy.csv" for dataset in DATASETS}
    inputs = [summary_path, position_path, *trial_paths.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing preparation input: {path}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    marker = AUDIT_DIR / "completed_run.txt"
    marker.unlink(missing_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    hashes = {str(path.resolve()): file_hash(path) for path in inputs}
    summary = pd.read_csv(summary_path, encoding=CSV_ENCODING, keep_default_na=False,
                          float_precision="round_trip", dtype={key: str for key in KEYS})
    positions = pd.read_csv(position_path, encoding=CSV_ENCODING, keep_default_na=False,
                            float_precision="round_trip", dtype={key: str for key in KEYS},
                            usecols=["seed", *KEYS, "Position", "target_id", "target_logit", "deficit_z", "blocking_z"])
    measures, normalization = build_item_measures(summary, positions)
    del positions
    results, sources, coverage, missing = {}, {}, [], []
    for dataset, path in trial_paths.items():
        source = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
        sources[dataset] = source
        result, counts, absent = merge_trials(dataset, source, measures)
        results[dataset] = result
        coverage.append(counts)
        missing.append(absent)
    manifest = []
    with tempfile.TemporaryDirectory(prefix=".logit_measures_", dir=DATA_DIR) as directory:
        staged = Path(directory)
        for dataset, result in results.items():
            filename = f"{dataset}_trials_with_logit_measures.csv"
            path = staged / filename
            result.to_csv(path, index=False, encoding=CSV_ENCODING, float_format=CSV_FLOAT_FORMAT)
            # Original numeric spellings and blank fields survive serialization.
            restored = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
            pd.testing.assert_frame_equal(restored[sources[dataset].columns], sources[dataset], check_exact=True)
            for name in METRICS:
                values = pd.to_numeric(restored[name].replace("", np.nan), errors="raise")
                require(np.allclose(values, result[name], atol=ATOL, rtol=RTOL, equal_nan=True),
                        f"Trial metric serialization differs: {dataset}, {name}")
            manifest.append(dict(run_id=run_id, dataset=dataset, path=(DATA_DIR / filename).relative_to(ROOT).as_posix(),
                                 md5=file_hash(path, "md5"), sha256=file_hash(path)))
        require(all(file_hash(Path(path)) == digest for path, digest in hashes.items()),
                "Inputs changed during preparation; rerun with stable inputs.")
        for row in manifest:
            destination = Path(row["path"])
            os.replace(staged / destination.name, destination)
            print(f"Saved {destination}")
    pd.DataFrame(coverage).to_csv(AUDIT_DIR / "coverage.csv", index=False, encoding=CSV_ENCODING)
    pd.concat(missing, ignore_index=True).to_csv(AUDIT_DIR / "missing_items.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame(manifest).to_csv(AUDIT_DIR / "output_manifest.csv", index=False, encoding=CSV_ENCODING)
    metadata = dict(run_id=run_id, hidden_dim=HIDDEN_DIM, epoch=EPOCH, seeds=SEEDS,
                    ddof=STANDARD_DEVIATION_DDOF, input_sha256=hashes, outputs=manifest,
                    target_normalization=normalization,
                    aggregation="Seed-wise position z-score; within Character-Word positions; equal seed mean",
                    formulas={"deficit_measure": "target - mean(non-target)",
                              "blocking_measure": "target - max(non-target)",
                              "target_logit_measure": "target"})
    (AUDIT_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.write_text(run_id + "\n", encoding="utf-8")
    print(f"Preparation complete: {run_id}. Review {AUDIT_DIR / 'coverage.csv'} before fitting.")


if __name__ == "__main__":
    main()
