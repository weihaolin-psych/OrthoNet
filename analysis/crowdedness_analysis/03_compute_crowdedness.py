"""
03_compute_crowdedness.py  (compute only)
=========================================
Compute crowdedness for every (character, word) target token across the FULL
lambda grid, in both embedding spaces (semantic, sound) and both versions
(raw = unweighted, freq = frequency-weighted).

This script does the expensive part only -- cosine distances + the lambda sweep
of crowdedness values. NO correlation and NO optimal-lambda selection happen
here; that lives in 04 (cheap, re-runnable without recomputing distances).

    raw  crowdedness = Sum exp(-lambda * d_cos)
    freq crowdedness = Sum log(f+1) * exp(-lambda * d_cos)

Targets: union of the three trial files (Huang, Wang Small, Wang Full). These
cover the Wang 1400 selection tokens too (Wang 1400 is a subset of Wang Full).

Output: analysis/crowdedness_analysis/output/crowdedness_all_lambdas.csv
    long over lambda; cols: Character, Word, lam, sem_raw, sem_freq, snd_raw, snd_freq
"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import time, os
import warnings

warnings.filterwarnings("ignore")

# ── Configuration (paths relative to the repo root) ──────────────────────────
SEM_EMBEDDING_PATH   = "data/embeddings/word_meaning_embeddings.pt"
SOUND_EMBEDDING_PATH = "data/embeddings/word_pinyin_embeddings.pt"
SEM_EMB_FIELD, SOUND_EMB_FIELD = "meaning_embedding", "pinyin_embedding"

TARGET_FILES = [
    "data/processed/wang_full_trials.csv",
    "data/processed/wang_small_trials.csv",
    "data/processed/huang_trials.csv",
]

OUTPUT_DIR     = "analysis/crowdedness_analysis/output"
ALL_LAMBDAS_CSV = os.path.join(OUTPUT_DIR, "crowdedness_all_lambdas.csv")

LAMBDA_RANGE = np.concatenate([[0], np.logspace(-4, 2, 30)])


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_keys(df):
    return df.rename(columns={"Target Char": "Character", "Context Word": "Word",
                              "char": "Character", "word": "Word",
                              "character": "Character"})

def load_and_flatten_pt(pt_path, emb_field):
    print(f"  Loading: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    chars, words, embs, freqs = [], [], [], []
    for word, wi in data.items():
        wf = wi.get("frequency", 1.0)
        for ci in wi["characters"]:
            if emb_field not in ci:
                continue
            emb = ci[emb_field]
            vec = (emb.detach().cpu().numpy().astype(np.float64)
                   if isinstance(emb, torch.Tensor) else np.array(emb, dtype=np.float64))
            if vec.ndim > 1:
                vec = vec.flatten()
            chars.append(ci["char"]); words.append(word); embs.append(vec); freqs.append(wf)
    embeddings = np.stack(embs)
    log_freqs  = np.log(np.array(freqs, dtype=np.float64) + 1)
    print(f"    Tokens: {len(chars)}, dim: {embeddings.shape[1]}, unique chars: {len(set(chars))}")
    return chars, words, embeddings, log_freqs

def build_lookups(chars, words):
    cw, ci = {}, {}
    for idx, (ch, w) in enumerate(zip(chars, words)):
        cw.setdefault((ch, w), idx); ci.setdefault(ch, []).append(idx)
    return cw, {ch: np.asarray(v, dtype=np.int64) for ch, v in ci.items()}

def precompute_distances(targets, chars, embeddings, log_freqs, cw, ci, label=""):
    n = len(chars); dl, fl = [], []; t0 = time.time(); mask_cache = {}
    for i, (tc, tw) in enumerate(targets):
        if (tc, tw) not in cw:
            dl.append(None); fl.append(None); continue
        ti = cw[(tc, tw)]; te = embeddings[ti:ti+1]
        if tc not in mask_cache:
            m = np.ones(n, dtype=bool)
            same = ci.get(tc)
            if same is not None:
                m[same] = False
            mask_cache[tc] = m
        cm = mask_cache[tc]
        dl.append(cdist(te, embeddings[cm], metric="cosine")[0]); fl.append(log_freqs[cm])
        if (i+1) % 300 == 0 or i+1 == len(targets):
            print(f"    {label}: {i+1}/{len(targets)} ({time.time()-t0:.1f}s)")
    return dl, fl

def crowdedness_from_dists(dl, lam, fl=None):
    out = np.full(len(dl), np.nan)
    for i, d in enumerate(dl):
        if d is None:
            continue
        out[i] = (np.sum(fl[i]) if fl else len(d)) if lam == 0 else \
                 (np.sum(fl[i] * np.exp(-lam * d)) if fl else np.sum(np.exp(-lam * d)))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60 + "\n1. TARGET TOKENS\n" + "=" * 60)
    frames = []
    for path in TARGET_FILES:
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}"); continue
        d = normalize_keys(pd.read_csv(path))[["Character", "Word"]].drop_duplicates()
        print(f"  {path}: {len(d)} tokens"); frames.append(d)
    df_targets = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    targets = list(zip(df_targets["Character"], df_targets["Word"]))
    print(f"  Target universe: {len(df_targets)} tokens")

    print("\n" + "=" * 60 + "\n2. EMBEDDINGS + DISTANCES\n" + "=" * 60)
    print("\nSemantic:")
    sc, sw, se, slf = load_and_flatten_pt(SEM_EMBEDDING_PATH, SEM_EMB_FIELD)
    s_cw, s_ci = build_lookups(sc, sw)
    sem_dl, sem_fl = precompute_distances(targets, sc, se, slf, s_cw, s_ci, "SEM")
    print("\nSound:")
    nc, nw, ne, nlf = load_and_flatten_pt(SOUND_EMBEDDING_PATH, SOUND_EMB_FIELD)
    n_cw, n_ci = build_lookups(nc, nw)
    snd_dl, snd_fl = precompute_distances(targets, nc, ne, nlf, n_cw, n_ci, "SND")

    print("\n" + "=" * 60 + "\n3. CROWDEDNESS OVER LAMBDA GRID\n" + "=" * 60)
    parts = []
    t0 = time.time()
    for j, lam in enumerate(LAMBDA_RANGE):
        part = pd.DataFrame({
            "Character": df_targets["Character"],
            "Word":      df_targets["Word"],
            "lam":       lam,
            "sem_raw":   crowdedness_from_dists(sem_dl, lam),
            "sem_freq":  crowdedness_from_dists(sem_dl, lam, fl=sem_fl),
            "snd_raw":   crowdedness_from_dists(snd_dl, lam),
            "snd_freq":  crowdedness_from_dists(snd_dl, lam, fl=snd_fl),
        })
        parts.append(part)
        print(f"    lam={lam:.4g} done ({j+1}/{len(LAMBDA_RANGE)}, {time.time()-t0:.1f}s)")

    df_all = pd.concat(parts, ignore_index=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_all.to_csv(ALL_LAMBDAS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Saved {ALL_LAMBDAS_CSV}  "
          f"({len(df_all)} rows = {len(df_targets)} tokens x {len(LAMBDA_RANGE)} lambdas)")
    print("\nDone. Run 04 for partial correlation, selection, and the figure.")


if __name__ == "__main__":
    main()
