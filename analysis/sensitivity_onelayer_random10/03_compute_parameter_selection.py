"""Compute seed-wise partial correlations and select hidden dimension/epoch.

Run from the repository root: python analysis/sensitivity_onelayer_random10/03_compute_parameter_selection.py
Reads the supplied covariate-merged grid (or an explicitly rebuilt grid).
Writes per-seed correlations, averaged grids, selected parameters and selected
cell correlations to output/. No plotting, training or model evaluation.
"""
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually.")

# ---- Configuration ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = "onelayer"
ANALYSIS_NAME = f"sensitivity_{ARCHITECTURE}_random10"
INPUT_DIR = ROOT / "data/analysis_inputs" / ANALYSIS_NAME
OUTPUT_DIR = ROOT / "analysis" / ANALYSIS_NAME / "output"
GRID_FILE = "wang_1400_entropy_with_covariates.csv"
SEEDS = [7, 42, 88, 123, 200, 314, 456, 555, 789, 999]
COVARIATES = ["FreqCount", "AoA", "zwFamiliarity", "SRO", "zRegularity", "nStroke"]
OUTCOME = "Log_Latency"
CORRELATION_FLOAT_FORMAT = "%.17g"


# ---- Residualize both variables using the same covariates and intercept -----
def partial_corr(group):
    covariates = group[COVARIATES].to_numpy()
    x = group["Entropy"].to_numpy()
    y = group[OUTCOME].to_numpy()
    residual_x = x - LinearRegression().fit(covariates, x).predict(covariates)
    residual_y = y - LinearRegression().fit(covariates, y).predict(covariates)
    return pearsonr(residual_x, residual_y)


# ---- Match the original complete-case selection and mean of seed-wise r -----
def main():
    rebuilt = OUTPUT_DIR / GRID_FILE
    source = rebuilt if rebuilt.is_file() else INPUT_DIR / GRID_FILE
    if not source.is_file():
        raise SystemExit(f"Missing input: {source}. Merge the companion Zenodo data package first.")
    frame = pd.read_csv(source)
    required = ["Seed", "hidden_dim", "Epoch", "Entropy", "TOP", OUTCOME, *COVARIATES]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing grid columns: {sorted(missing)}")
    frame = frame.dropna(subset=COVARIATES + ["Entropy", "TOP", OUTCOME])
    if set(frame.Seed.unique()) != set(SEEDS):
        raise ValueError("The grid must contain the ten configured seeds.")
    records = []
    for seed in SEEDS:
        for (hidden, epoch), group in frame.loc[frame.Seed.eq(seed)].groupby(["hidden_dim", "Epoch"]):
            r, p = partial_corr(group)
            records.append(dict(Seed=seed, hidden_dim=hidden, Epoch=epoch, n=len(group), r=r, p=p))
    per_seed = pd.DataFrame(records)
    coverage = per_seed.groupby(["hidden_dim", "Epoch"]).Seed.nunique()
    if not coverage.eq(len(SEEDS)).all() or not np.isfinite(per_seed[["r", "p"]]).all().all():
        raise ValueError("Incomplete or nonfinite seed-wise correlation grid.")
    # pandas std uses the original sample-SD convention, ddof=1.
    averaged = per_seed.groupby(["hidden_dim", "Epoch"]).agg(
        r_mean=("r", "mean"), r_std=("r", "std")).reset_index()
    best = averaged.loc[averaged.r_mean.idxmax()]
    hidden, epoch = int(best.hidden_dim), int(best.Epoch)
    optimal = pd.DataFrame([dict(hidden_dim=hidden, epoch=epoch,
                                r_mean=best.r_mean, r_std=best.r_std)])
    selected = per_seed.loc[per_seed.hidden_dim.eq(hidden) & per_seed.Epoch.eq(epoch),
                            ["Seed", "n", "r", "p"]].sort_values("Seed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {"partial_correlations_by_seed.csv": per_seed,
               "partial_correlations_averaged.csv": averaged,
               "optimal_parameters.csv": optimal,
               "optimal_cell_correlations.csv": selected}
    for name, result in outputs.items():
        result.to_csv(OUTPUT_DIR / name, index=False, float_format=CORRELATION_FLOAT_FORMAT)
    print(optimal.to_string(index=False))


if __name__ == "__main__":
    main()
