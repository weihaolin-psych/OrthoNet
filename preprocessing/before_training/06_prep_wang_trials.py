
import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Input
WANG_TRIAL_XLSX = "data/raw/wang2020/1.Latency & Duration & ACC.xlsx"
WANG_DB_XLSX    = "data/raw/wang2020/Database.xlsx"
HUANG_XLSX      = "data/raw/huang2021/huanglxwc_tip_of_pen.xlsx"  # for wang_small character list

# Output
OUTPUT_WANG_FULL  = "data/processed/wang_full_trials.csv"
OUTPUT_WANG_SMALL = "data/processed/wang_small_trials.csv"

# Response code column (Chinese header in raw file)
RESP_COL = "判断(0-正确；2-TOP）"

# ==========================================
# 1. Load trial-level behavioral data
# ==========================================
print("Step 1: Loading Wang trial-level data...")
df_trial = pd.read_excel(WANG_TRIAL_XLSX)
print(f"  Trials: {len(df_trial)} ({df_trial['被试'].nunique()} subjects × {df_trial['Item'].nunique()} items)")
print(f"  Response counts: {df_trial[RESP_COL].value_counts().sort_index().to_dict()}")

# ==========================================
# 2. Load character-level data (covariates + word mapping)
# ==========================================
print("Step 2: Loading character-level data from Database.xlsx...")
df_char = pd.read_excel(WANG_DB_XLSX, sheet_name="Data")
print(f"  Characters: {len(df_char)}")

# ==========================================
# 3. Merge trial data with character-level covariates
# ==========================================
print("Step 3: Merging trial data with covariates...")

# Select covariates from Database.xlsx
char_cols = [
    "Item", "Character", "Word",
    "FreqCount", "AoA", "zwFamiliarity", "SRO", "zRegularity", "nStroke",
]
df_char_sub = df_char[char_cols].copy()

df = df_trial.merge(df_char_sub, on=["Item", "Character"], how="left")

n_missing = df["Word"].isna().sum()
if n_missing > 0:
    print(f"  WARNING: {n_missing} trials missing character-level data!")
else:
    print(f"  All {len(df)} trials matched.")


# ==========================================
# 4. Create analysis variables
# ==========================================
print("Step 4: Creating analysis variables...")

# 4a. TOP binary: code 0 = correct (TOP=0), code 2 = TOP (TOP=1)
df["top_binary"] = np.where(df[RESP_COL] == 0, 0,
                   np.where(df[RESP_COL] == 2, 1, np.nan))
df["top_eligible"] = df[RESP_COL].isin([0, 2])

# 4b. RT: correct trials with Latency > 0
df["log_rt"] = np.where(df["Latency"] > 0, np.log(df["Latency"]), np.nan)
df["rt_eligible"] = (df[RESP_COL] == 0) & (df["Latency"] > 0)

# 4c. Rename columns
df = df.rename(columns={
    "被试": "subject",
    "Latency": "rt",
    "FreqCount": "log_freq",
    "AoA": "aoa",
    "zwFamiliarity": "context_familiarity",
    "SRO": "phonetic_radical_order",
    "zRegularity": "regularity",
    "nStroke": "n_stroke",
    "性别": "gender",
    "Item": "item",
})

# Recode gender: raw data has 1/2, recode to labels
df["gender"] = df["gender"].map({1: "M", 2: "F"})

# 4d. Map response codes to readable labels
resp_map = {0: "CW", 1: "Misspelt", 2: "TOP", 3: "Misremembered", 4: "Misheard"}
df["response"] = df[RESP_COL].map(resp_map)

# 4e. Select columns
output_cols = [
    "subject", "item", "Character", "Word",
    "response", "top_binary", "top_eligible",
    "rt", "log_rt", "rt_eligible",
    "log_freq", "aoa", "context_familiarity",
    "phonetic_radical_order", "regularity", "n_stroke",
    "gender",
]
df_out = df[output_cols].copy()

# ==========================================
# 5. Save wang_full
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_WANG_FULL), exist_ok=True)
df_out.to_csv(OUTPUT_WANG_FULL, index=False, encoding="utf-8-sig")

# ==========================================
# 6. Create and save wang_small (200 Huang characters)
# ==========================================
print("Step 6: Creating wang_small (200 Huang characters)...")
df_huang_mat = pd.read_excel(HUANG_XLSX, sheet_name="Materials")
huang_chars = set(df_huang_mat[df_huang_mat["Item"] <= 200]["Character"].unique())

df_small = df_out[df_out["Character"].isin(huang_chars)].copy()
df_small.to_csv(OUTPUT_WANG_SMALL, index=False, encoding="utf-8-sig")

# ==========================================
# 7. Summary
# ==========================================
def print_summary(name, df):
    print(f"\n  --- {name} ---")
    print(f"  Total trials:    {len(df)}")
    print(f"  Subjects:        {df['subject'].nunique()}")
    print(f"  Characters:      {df['Character'].nunique()}")
    print(f"  Response: {df['response'].value_counts().to_dict()}")
    top_df = df[df["top_eligible"]]
    if len(top_df) > 0:
        print(f"  TOP trials:      {len(top_df)} (TOP rate = {top_df['top_binary'].mean():.3f})")
    rt_df = df[df["rt_eligible"]]
    if len(rt_df) > 0:
        print(f"  RT trials:       {len(rt_df)} (mean RT = {rt_df['rt'].mean():.1f} ms)")

print(f"\n{'='*60}")
print(f"OUTPUTS")
print(f"{'='*60}")
print(f"  {OUTPUT_WANG_FULL}")
print(f"  {OUTPUT_WANG_SMALL}")
print_summary("Wang Full (1600 chars)", df_out)
print_summary("Wang Small (200 chars)", df_small)
print(f"{'='*60}")
print("✅ Done!")
