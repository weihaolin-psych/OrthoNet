"""
Step 2b: Merge Lexical Covariates into Wang 1400 Entropy Data
==============================================================

Adds the 6 lexical predictors from Wang Database to the entropy file,
so that partial correlation analysis can be done from a single CSV.

Covariates (Huang 2021 significant predictors of TOP):
    - FreqCount       : log character frequency
    - AoA             : age of acquisition
    - zwFamiliarity   : context word familiarity (z-scored)
    - SRO             : phonetic radical order (0/1)
    - zRegularity     : spelling regularity (z-scored)
    - nStroke         : number of strokes

Input:
    - wang_1400_entropy_all.csv   : from Step 2
    - Database.xlsx               : Wang et al. (2020)

Output:
    - wang_1400_entropy_with_covariates.csv
"""

import pandas as pd
import os
import numpy as np

# ============================================================
# Paths
# ============================================================
ENTROPY_CSV   = "analysis/sensitivity_onelayer_random10/output/wang_1400_entropy_all.csv"
WANG_DATABASE = "data/raw/wang2020/Database.xlsx"
OUTPUT_CSV    = "analysis/sensitivity_onelayer_random10/output/wang_1400_entropy_with_covariates.csv"
SUPPLIED_GRID = "data/analysis_inputs/sensitivity_onelayer_random10/wang_1400_entropy_with_covariates.csv"

COVARIATES = ["FreqCount", "AoA", "zwFamiliarity", "SRO", "zRegularity", "nStroke"]

# ============================================================
# Load and merge
# ============================================================
print("Loading entropy data...")
# Use the canonical merged grid supplied with the archive when the
# raw extraction has not been regenerated. Rebuild with script02a
# first when a new covariate merge is required.
if not os.path.isfile(ENTROPY_CSV):
    if os.path.isfile(SUPPLIED_GRID):
        print("Using the supplied covariate-merged grid: " + SUPPLIED_GRID)
        print("To rebuild it from model evaluations, run script02a first.")
        raise SystemExit(0)
    raise FileNotFoundError("Missing entropy grid; run script02a to extract model evaluations first.")

df_entropy = pd.read_csv(ENTROPY_CSV)
print(f"  {len(df_entropy)} rows, {df_entropy[['Character','Word']].drop_duplicates().shape[0]} chars")
print(f"  Seeds: {sorted(df_entropy['Seed'].unique())}")

print("Loading Wang Database covariates...")
df_wang = pd.read_excel(WANG_DATABASE, sheet_name="Data")
df_covars = df_wang[["Character", "Word"] + COVARIATES].copy()
print(f"  {len(df_covars)} chars, covariates: {COVARIATES}")

print("Merging...")
df = df_entropy.merge(df_covars, on=["Character", "Word"], how="left")

na_rows = df[COVARIATES].isna().any(axis=1).sum()
if na_rows > 0:
    print(f"  WARNING: {na_rows} rows with missing covariates")
    missing = df[df[COVARIATES].isna().any(axis=1)][["Character", "Word"]].drop_duplicates()
    print(f"  Missing chars: {missing.to_dict('records')[:10]}")
else:
    print(f"  All covariates merged successfully, no missing values.")

# log-transform latency
col_idx = df.columns.get_loc("Latency_Correct") + 1
df.insert(col_idx, "Log_Latency", np.log(df["Latency_Correct"]))

# save
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n→ Saved {OUTPUT_CSV}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  File size: {os.path.getsize(OUTPUT_CSV)/1024/1024:.1f} MB")
print("✅ Done!")
