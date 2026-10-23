"""
04_select_lambda.py
==================================
Read the all-lambda crowdedness (from 03), compute the partial correlation of
each crowdedness measure with Wang 1400 writing latency (controlling type
frequency), select the optimal lambda per measure,
and write the optimal-lambda crowdedness for the merge step (05).

Everything you might want to tweak -- which crowdedness columns to analyze,
which latency column, which control variable, whether to log crowdedness -- is
in ANALYSIS CONFIG below.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

# ══════════════════════════════════════════════════════════════
#  ANALYSIS CONFIG  (edit these to change what is analyzed)
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR = "analysis/crowdedness_analysis/output"

# Inputs
ALL_LAMBDAS_CSV = os.path.join(OUTPUT_DIR, "crowdedness_all_lambdas.csv")  # from 03
WANG_1400_CSV   = os.path.join(OUTPUT_DIR, "wang_1400_characters.csv")     # latency
TYPE_FREQ_CSV   = "data/processed/type_frequency.csv"                       # control

# Which crowdedness columns (in the all-lambda file) to analyze.
# Each name must be "<space>_<version>"; space in {sem, snd}, version in {raw, freq}.
CROWD_COLS  = ["sem_raw", "sem_freq", "snd_raw", "snd_freq"]

LATENCY_COL = "Log_Latency"      # human DV to correlate against (already logged in 01)
CONTROL_COL = "log_type_frequency"   # control variable (un-logged -> raw curve hits ~0 at lam=0)
USE_LOG_CROWD = True             # log-transform crowdedness before correlating?

# Outputs
SWEEP_CSV   = os.path.join(OUTPUT_DIR, "crowdedness_sweep_latency.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "crowdedness_optimal_lambda_summary.csv")
OPTIMAL_CSV = os.path.join(OUTPUT_DIR, "crowdedness_optimal_lambda.csv")   # feeds 05


# Map analyzed columns -> output column names expected by 05 (note snd -> sound)
MEAS2OUT = {
    "sem_raw":  "sem_crowdedness_raw",   "sem_freq": "sem_crowdedness_freq",
    "snd_raw":  "sound_crowdedness_raw", "snd_freq": "sound_crowdedness_freq",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_keys(df):
    return df.rename(columns={"Target Char": "Character", "Context Word": "Word",
                              "char": "Character", "word": "Word",
                              "character": "Character"})

def partial_r(x, y, z):
    """Partial corr of x and y controlling z. If x is (near) perfectly linear in
    z -- as raw crowdedness is in type frequency at lambda=0 -- the residual has
    no variance, so the partial correlation is defined as 0."""
    xr = x - np.polyval(np.polyfit(z, x, 1), z)
    yr = y - np.polyval(np.polyfit(z, y, 1), z)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return 0.0, 1.0
    return pearsonr(xr, yr)


# ── Load + merge ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(ALL_LAMBDAS_CSV)

    sel = normalize_keys(pd.read_csv(WANG_1400_CSV))[["Character", "Word", LATENCY_COL]]
    sel = sel.dropna(subset=[LATENCY_COL]).drop_duplicates(["Character", "Word"])

    tf = normalize_keys(pd.read_csv(TYPE_FREQ_CSV))[["Character", CONTROL_COL]] \
        .drop_duplicates("Character")

    df = df.merge(sel, on=["Character", "Word"], how="left")  # adds LATENCY_COL (NaN if not a 1400 token)
    df = df.merge(tf, on="Character", how="left")             # adds CONTROL_COL
    return df


# ── Sweep: partial r per measure per lambda (selection subset) ───────────────
def run_sweep(df):
    sel = df[df[LATENCY_COL].notna() & df[CONTROL_COL].notna()].copy()
    rows = []
    for col in CROWD_COLS:
        space, version = col.split("_")
        for lam, g in sel.groupby("lam"):
            x, y, z = g[col].to_numpy(), g[LATENCY_COL].to_numpy(), g[CONTROL_COL].to_numpy()
            ok = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
            if ok.sum() < 10:
                r = np.nan
            else:
                xx = np.log(x[ok] + 1e-30) if USE_LOG_CROWD else x[ok]
                r, _ = partial_r(xx, y[ok], z[ok])
            rows.append(dict(measure=col, space=space, version=version,
                             lam=float(lam), r=r, n=int(ok.sum())))
    return pd.DataFrame(rows)


def pick_optimal(sweep, measure):
    s = sweep[(sweep["measure"] == measure) & (sweep["lam"] > 0)]
    row = s.loc[s["r"].idxmax()]
    return float(row["lam"]), float(row["r"])


# ── Numerical output ─────────────────────────────────────────────────────────────────────



# ── Extract optimal-lambda crowdedness for the merge step ────────────────────
def build_optimal_output(df, optima):
    out = df[["Character", "Word"]].drop_duplicates().reset_index(drop=True)
    lambdas = {}
    for col in CROWD_COLS:
        lam_opt, _ = optima[col]
        lambdas[col] = lam_opt
        sl = df[np.isclose(df["lam"], lam_opt)][["Character", "Word", col]] \
            .rename(columns={col: MEAS2OUT[col]})
        out = out.merge(sl, on=["Character", "Word"], how="left")
    for col, outname in MEAS2OUT.items():
        if outname in out.columns:
            out["log_" + outname] = np.log(out[outname] + 1e-30)
    out["lambda_sem_raw"]    = lambdas.get("sem_raw")
    out["lambda_sem_freq"]   = lambdas.get("sem_freq")
    out["lambda_sound_raw"]  = lambdas.get("snd_raw")
    out["lambda_sound_freq"] = lambdas.get("snd_freq")
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading all-lambda crowdedness + latency + control...")
    df = load_data()
    n_sel = df[df[LATENCY_COL].notna()]["Character"].nunique()
    print(f"  Selection tokens with latency: ~{n_sel} characters")

    print("Running partial-correlation sweep...")
    sweep = run_sweep(df)

    optima = {col: pick_optimal(sweep, col) for col in CROWD_COLS}
    print("\nOptimal lambda per measure:")
    for col, (lam, r) in optima.items():
        print(f"  {col:9s}: lambda={lam:.4g}  partial r={r:.4f}")


    out = build_optimal_output(df, optima)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sweep.to_csv(SWEEP_CSV, index=False)
    pd.DataFrame([{"measure": c, "lambda": optima[c][0], "partial_r": optima[c][1]}
                  for c in CROWD_COLS]).to_csv(SUMMARY_CSV, index=False)
    out.to_csv(OPTIMAL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Saved {SWEEP_CSV}")
    print(f"  Saved {SUMMARY_CSV}")
    print(f"  Saved {OPTIMAL_CSV}  ({len(out)} tokens)  -> feeds 05")
    print("\nDone.")


if __name__ == "__main__":
    main()
