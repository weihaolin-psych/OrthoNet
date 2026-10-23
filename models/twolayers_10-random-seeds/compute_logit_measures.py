"""Compute context-specific relative target support from complete logit archives.

Run from the project root: python -u compute_logit_measures.py
Reads ten hidden22/epoch11 NPZs from models/twolayers_10-random-seeds/full_logit/.
Writes only position_metrics_by_seed.csv and character_word_measures.csv to
data/logit_measures/. Requires NumPy and pandas; never trains,
fits behavioral models, plots, or modifies source archives.
"""

import argparse
import os
import tempfile
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually in the analysis environment.")

# ---- Configuration ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models/twolayers_10-random-seeds"
OUTPUT_DIR = PROJECT_ROOT / "data/logit_measures"
HIDDEN_DIM = 22
EPOCH = 11
SEEDS = (7, 42, 88, 123, 200, 314, 456, 555, 789, 999)
EXPECTED_POSITIONS = 106_844
EXPECTED_CANDIDATES = 4_677
EXPECTED_CHARACTER_WORDS = 105_617
CHUNK_ROWS = 1024
STANDARD_DEVIATION_DDOF = 1
CHECK_ATOL = 1e-10
CHECK_RTOL = 1e-12
CSV_ENCODING = "utf-8-sig"
CSV_FLOAT_FORMAT = "%.17g"
POSITION_FILENAME = "position_metrics_by_seed.csv"
SUMMARY_FILENAME = "character_word_measures.csv"
INDEX_FIELDS = ("candidate_chars", "context_words", "target_chars", "positions", "target_ids")
POSITION_COLUMNS = ["seed", "Character", "Word", "Position", "target_id", "target_logit",
                    "mean_non_target_logit", "max_non_target_logit", "deficit_raw",
                    "blocking_raw", "deficit_z", "blocking_z"]
SUMMARY_COLUMNS = ["Character", "Word", "deficit_measure", "blocking_measure",
                   "n_positions", "n_seeds"]
KEYS = ["Character", "Word"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


# ---- Validate archive identity before computing any metrics -----------------
def validate_archive(archive, seed, reference_indices):
    required = {"logits", "hidden_dim", "epoch", "seed", *INDEX_FIELDS}
    require(required.issubset(archive.files), f"Missing NPZ fields for seed {seed}.")
    for name, expected in (("hidden_dim", HIDDEN_DIM), ("epoch", EPOCH), ("seed", seed)):
        value = archive[name]
        require(value.shape == () and value.dtype.kind in "iu" and value.item() == expected,
                f"Incorrect NPZ model identifier: {name}, seed {seed}.")
    indices = {name: archive[name] for name in INDEX_FIELDS}
    candidates = indices["candidate_chars"]
    require(candidates.shape == (EXPECTED_CANDIDATES,) and candidates.dtype.kind == "U",
            f"Invalid candidate labels for seed {seed}.")
    require(np.all(np.char.str_len(candidates) == 1) and
            len(np.unique(candidates)) == EXPECTED_CANDIDATES,
            f"Candidate characters must be unique single characters: seed {seed}.")
    for name in ("context_words", "target_chars"):
        value = indices[name]
        require(value.shape == (EXPECTED_POSITIONS,) and value.dtype.kind == "U" and
                np.all(np.char.str_len(value) > 0), f"Invalid NPZ labels: {name}, seed {seed}.")
    for name in ("positions", "target_ids"):
        value = indices[name]
        require(value.shape == (EXPECTED_POSITIONS,) and value.dtype.kind in "iu" and
                np.all(value >= 0), f"Invalid NPZ indices: {name}, seed {seed}.")
    target_ids = indices["target_ids"]
    require(np.all(target_ids < EXPECTED_CANDIDATES), f"Target index out of range: seed {seed}.")
    require(np.array_equal(candidates[target_ids], indices["target_chars"]),
            f"Target character/column mismatch: seed {seed}.")
    require(len(np.unique(indices["target_chars"])) == EXPECTED_CANDIDATES,
            f"Incomplete target inventory: seed {seed}.")
    row_keys = pd.DataFrame({"Word": indices["context_words"], "Position": indices["positions"]})
    require(not row_keys.duplicated().any(), f"Duplicate original input positions: seed {seed}.")
    if reference_indices is not None:
        for name in INDEX_FIELDS:
            require(np.array_equal(indices[name], reference_indices[name]),
                    f"Cross-seed row/column order mismatch: {name}, seed {seed}.")
    return indices


# ---- Compute raw differences first, then standardize across input positions -
def z_score(values, description, report=True):
    values = np.asarray(values, dtype=np.float64)
    require(len(values) > STANDARD_DEVIATION_DDOF and np.isfinite(values).all(),
            f"Invalid normalization input: {description}.")
    mean = values.mean()
    sd = values.std(ddof=STANDARD_DEVIATION_DDOF)
    require(np.isfinite(sd) and sd > 0, f"Zero/invalid standard deviation: {description}.")
    standardized = (values - mean) / sd
    require(np.isfinite(standardized).all() and
            np.isclose(standardized.mean(), 0, atol=CHECK_ATOL, rtol=0) and
            np.isclose(standardized.std(ddof=STANDARD_DEVIATION_DDOF), 1,
                       atol=CHECK_ATOL, rtol=CHECK_RTOL),
            f"Z-score validation failed: {description}.")
    if report:
        print(f"{description}: n={len(values)}, mean={mean:.17g}, sd={sd:.17g}, "
              f"ddof={STANDARD_DEVIATION_DDOF}; z_mean={standardized.mean():.3g}, "
              f"z_sd={standardized.std(ddof=STANDARD_DEVIATION_DDOF):.17g}", flush=True)
    return standardized


def compute_position_metrics(logits, indices, seed):
    require(logits.dtype == np.float32 and
            logits.shape == (EXPECTED_POSITIONS, EXPECTED_CANDIDATES),
            f"Unexpected logit shape/dtype for seed {seed}: {logits.shape}, {logits.dtype}.")
    target = np.empty(EXPECTED_POSITIONS, dtype=np.float64)
    non_target_mean = np.empty_like(target)
    non_target_max = np.empty_like(target)
    for start in range(0, EXPECTED_POSITIONS, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, EXPECTED_POSITIONS)
        block = logits[start:stop]
        require(np.isfinite(block).all(), f"Nonfinite logits: seed {seed}, row {start}.")
        rows = np.arange(stop - start)
        columns = indices["target_ids"][start:stop]
        target[start:stop] = block[rows, columns]
        # Float64 accumulation avoids summing thousands of float32 values at
        # float32 precision. The target is explicitly removed from the mean.
        non_target_mean[start:stop] = (
            block.sum(axis=1, dtype=np.float64) - target[start:stop]
        ) / (EXPECTED_CANDIDATES - 1)
        competitors = block.copy()
        competitors[rows, columns] = -np.inf
        non_target_max[start:stop] = competitors.max(axis=1)

    deficit = target - non_target_mean
    blocking = target - non_target_max
    return pd.DataFrame({
        "seed": seed, "Character": indices["target_chars"], "Word": indices["context_words"],
        "Position": indices["positions"], "target_id": indices["target_ids"],
        "target_logit": target, "mean_non_target_logit": non_target_mean,
        "max_non_target_logit": non_target_max, "deficit_raw": deficit, "blocking_raw": blocking,
        "deficit_z": z_score(deficit, f"seed {seed}, deficit"),
        "blocking_z": z_score(blocking, f"seed {seed}, blocking"),
    }, columns=POSITION_COLUMNS)


# ---- Persist full precision and validate the CSV used for aggregation -------
def read_position_csv(path, reference_indices):
    dtypes = {"seed": "int64", "Character": str, "Word": str,
              "Position": "int64", "target_id": "int64"}
    dtypes.update({name: "float64" for name in POSITION_COLUMNS[5:]})
    frame = pd.read_csv(path, encoding=CSV_ENCODING, dtype=dtypes,
                        keep_default_na=False, float_precision="round_trip")
    require(frame.columns.tolist() == POSITION_COLUMNS, "Position CSV columns differ.")
    require(len(frame) == EXPECTED_POSITIONS * len(SEEDS), "Position CSV row count differs.")
    require(set(frame.seed.unique()) == set(SEEDS), "Position CSV seed inventory differs.")
    require(not frame.duplicated(["seed", "Word", "Position"]).any(), "Duplicate position CSV keys.")
    numeric = POSITION_COLUMNS[5:]
    require(np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all(), "Nonfinite CSV metrics.")
    for seed in SEEDS:
        selected = frame.loc[frame.seed.eq(seed)]
        require(len(selected) == EXPECTED_POSITIONS, f"CSV seed row count differs: {seed}.")
        for column, field in (("Character", "target_chars"), ("Word", "context_words"),
                              ("Position", "positions"), ("target_id", "target_ids")):
            require(np.array_equal(selected[column].to_numpy(), reference_indices[field]),
                    f"CSV row identity mismatch: {column}, seed {seed}.")
        for name, component in (("deficit", "mean_non_target_logit"),
                                ("blocking", "max_non_target_logit")):
            require(np.allclose(selected[f"{name}_raw"], selected.target_logit - selected[component],
                                atol=CHECK_ATOL, rtol=CHECK_RTOL), f"CSV raw difference mismatch: {name}.")
            expected_z = z_score(selected[f"{name}_raw"].to_numpy(), f"CSV seed {seed}, {name}", report=False)
            require(np.allclose(selected[f"{name}_z"], expected_z, atol=CHECK_ATOL, rtol=CHECK_RTOL),
                    f"CSV normalization mismatch: {name}, seed {seed}.")
    return frame


def aggregate_character_words(frame):
    # Collapse repeated target positions only within the SAME context and seed.
    per_seed = frame.groupby(["seed", *KEYS], sort=True).agg(
        deficit_measure=("deficit_z", "mean"), blocking_measure=("blocking_z", "mean"),
        n_positions=("Position", "size"),
    ).reset_index()
    require(len(per_seed) == EXPECTED_CHARACTER_WORDS * len(SEEDS), "Character-Word coverage differs.")
    grouped = per_seed.groupby(KEYS, sort=True)
    require(grouped.n_positions.nunique().eq(1).all(), "Position multiplicity differs across seeds.")
    summary = grouped.agg(
        deficit_measure=("deficit_measure", "mean"), blocking_measure=("blocking_measure", "mean"),
        n_positions=("n_positions", "first"), n_seeds=("seed", "nunique"),
    ).reset_index()[SUMMARY_COLUMNS]
    require(len(summary) == EXPECTED_CHARACTER_WORDS and summary.n_seeds.eq(len(SEEDS)).all(),
            "Every Character-Word must have all ten seeds.")
    require(not summary.duplicated(KEYS).any(), "Duplicate summary keys.")
    require(summary.n_positions.sum() == EXPECTED_POSITIONS, "Summary position coverage differs.")
    return summary


# ---- One seed in memory; only a fully checked directory becomes visible -----
def build_csvs(input_dir, staging_dir):
    require(len(set(SEEDS)) == len(SEEDS), "Duplicate seeds in Configuration.")
    paths = {seed: input_dir / f"logits_hidden{HIDDEN_DIM}_epoch{EPOCH}_seed{seed}.npz" for seed in SEEDS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing NPZ inputs:\n" + "\n".join(missing))
    reference_indices = None
    position_path = staging_dir / POSITION_FILENAME
    # Keep one text stream open so the BOM/header occurs exactly once.
    with position_path.open("w", encoding=CSV_ENCODING, newline="") as stream:
        for number, (seed, path) in enumerate(paths.items()):
            print(f"Reading {path} ({path.stat().st_size} bytes)", flush=True)
            with np.load(path, allow_pickle=False) as archive:
                indices = validate_archive(archive, seed, reference_indices)
                if reference_indices is None:
                    reference_indices = indices
                logits = archive["logits"]
                frame = compute_position_metrics(logits, indices, seed)
                del logits  # NPZ members load into RAM; release before the next seed.
            frame.to_csv(stream, index=False, header=number == 0, float_format=CSV_FLOAT_FORMAT)
            del frame

    # The saved CSV, rather than an incidental in-memory precursor, is canonical.
    frame = read_position_csv(position_path, reference_indices)
    summary = aggregate_character_words(frame)
    summary_path = staging_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False, encoding=CSV_ENCODING, float_format=CSV_FLOAT_FORMAT)
    restored = pd.read_csv(summary_path, encoding=CSV_ENCODING,
                           dtype={"Character": str, "Word": str, "deficit_measure": "float64",
                                  "blocking_measure": "float64", "n_positions": "int64", "n_seeds": "int64"},
                           keep_default_na=False, float_precision="round_trip")
    pd.testing.assert_frame_equal(restored, summary, check_exact=True)
    print(f"Validated {len(frame)} position rows and {len(summary)} Character-Word rows.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=MODEL_ROOT / "full_logit")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="Must not exist; both CSVs are published together after validation.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(lock_fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        # TemporaryDirectory owns a wrapper; only its completed child is moved.
        with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as folder:
            staging_dir = Path(folder) / "complete"
            staging_dir.mkdir()
            build_csvs(args.input_dir, staging_dir)
            if output_dir.exists():
                raise FileExistsError(f"Output appeared during processing: {output_dir}")
            staging_dir.rename(output_dir)
        print(f"Verified and saved both CSVs: {output_dir}", flush=True)
    finally:
        lock_path.unlink()


if __name__ == "__main__":
    main()
