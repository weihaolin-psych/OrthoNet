
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Input
HUANG_XLSX      = "data/raw/huang2021/huanglxwc_tip_of_pen.xlsx"
SHEET_DATA      = "Data"
SHEET_MATERIALS = "Materials"

# Output
OUTPUT_FILE = "data/processed/huang_trials.csv"

# ==========================================
# 1. Load behavioral data
# ==========================================
print("Step 1: Loading Huang behavioral data...")
df_data = pd.read_excel(HUANG_XLSX, sheet_name=SHEET_DATA)
df_mat = pd.read_excel(HUANG_XLSX, sheet_name=SHEET_MATERIALS)

print(f"  Data sheet: {df_data.shape[0]} trials ({df_data['Subject'].nunique()} subjects × {df_data['Item'].nunique()} items)")
print(f"  Materials sheet: {df_mat.shape[0]} items")

# ==========================================
# 2. Merge item → character + context word
# ==========================================
print("Step 2: Merging item-character mapping...")
# Materials has 210 items; items 201-210 are practice items not in Data
df_mat = df_mat[df_mat["Item"] <= 200].copy()

df = df_data.merge(df_mat[["Item", "Character", "Word"]], on="Item", how="left")

# Verify merge
n_missing = df["Character"].isna().sum()
if n_missing > 0:
    print(f"  WARNING: {n_missing} trials missing character mapping!")
else:
    print(f"  All {len(df)} trials matched successfully.")


# ==========================================
# 3. Create analysis variables
# ==========================================
print("Step 3: Creating analysis variables...")

# 3a. TOP binary outcome: CW=0, PTOP/NTOP=1
#     Eligible trials: CW + PTOP + NTOP
#     Both PTOP and NTOP are TOP states (correctly selected target in forced-choice)
#     PTOP = TOP with partial orthographic access; NTOP = TOP without
#     This matches Huang 2021's reported TOP rate of ~19%
df["top_binary"] = np.where(df["Response"] == "CW", 0,
                   np.where(df["Response"].isin(["PTOP", "NTOP"]), 1, np.nan))
df["top_eligible"] = df["Response"].isin(["CW", "PTOP", "NTOP"])

# 3b. RT eligibility: CW trials with Latency > 0
df["log_rt"] = np.where(df["Latency"] > 0, np.log(df["Latency"]), np.nan)
df["rt_eligible"] = (df["Response"] == "CW") & (df["Latency"] > 0)

# 3c. Rename covariates for clarity
df = df.rename(columns={
    "Subject": "subject",
    "Item": "item",
    "Response": "response",
    "Latency": "rt",
    "FreqCount": "log_freq",          # already log-transformed in original data
    "AoA": "aoa",
    "zwFamiliarity": "context_familiarity",  # z-scored word familiarity
    "SRO": "phonetic_radical_order",   # 0 or 1 (recode to ±0.5 in analysis)
    "zRegularity": "regularity",       # z-scored spelling regularity
    "nStroke": "n_stroke",
    "Gender": "gender",
})

# 3d. Select and order columns for output
#     Keep only what's needed for mixed-effects models.
#     Everything else is always available in the raw xlsx.
output_cols = [
    # --- Trial identifiers ---
    "subject", "item", "Character", "Word",
    # --- Dependent variables ---
    "response", "top_binary", "top_eligible",
    "rt", "log_rt", "rt_eligible",
    # --- Covariates (Huang 2021 significant predictors) ---
    "log_freq", "aoa", "context_familiarity",
    "phonetic_radical_order", "regularity", "n_stroke",
    # --- Subject-level (included in Huang 2021 models) ---
    "gender",
]

df_out = df[output_cols].copy()

# ==========================================
# 4. Save
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# ==========================================
# 5. Summary
# ==========================================
print(f"\n{'='*60}")
print(f"OUTPUT: {OUTPUT_FILE}")
print(f"{'='*60}")
print(f"  Total trials:       {len(df_out)}")
print(f"  Subjects:           {df_out['subject'].nunique()}")
print(f"  Items:              {df_out['item'].nunique()}")
print(f"")
print(f"  --- Response breakdown ---")
print(f"  {df_out['response'].value_counts().to_string()}")
print(f"")
print(f"  --- TOP analysis (top_eligible=True) ---")
n_top = df_out.loc[df_out["top_eligible"], "top_binary"]
print(f"  Eligible trials:    {n_top.count()}")
print(f"  CW (TOP=0):         {(n_top == 0).sum()}")
print(f"  TOP (TOP=1):        {(n_top == 1).sum()} (PTOP + NTOP)")
print(f"  TOP rate:            {n_top.mean():.3f}")
print(f"")
print(f"  --- RT analysis (rt_eligible=True) ---")
rt_trials = df_out.loc[df_out["rt_eligible"], "log_rt"]
print(f"  Eligible trials:    {rt_trials.count()}")
print(f"  Mean RT (ms):       {df_out.loc[df_out['rt_eligible'], 'rt'].mean():.1f}")
print(f"  Mean log(RT):       {rt_trials.mean():.3f}")
print(f"")
print(f"{'='*60}")
print("✅ Done!")
