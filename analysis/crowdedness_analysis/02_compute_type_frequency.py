"""
02_compute_type_frequency.py
============================
Type frequency of a character = number of distinct words in the CLD that
contain it. Used as the (sole) control variable in the crowdedness lambda
selection, because crowdedness is mechanically tied to type frequency.

Input:  data/processed/word_centric_dict.json
Output: data/processed/type_frequency.csv
        cols: character, type_frequency, log_type_frequency
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
import os
from pyprojroot.here import here

WORD_DICT_PATH = here("data/processed/word_centric_dict.json")
OUTPUT_PATH    = here("data/processed/type_frequency.csv")


def main():
    print("Computing type frequency from CLD word dictionary...\n")
    with open(WORD_DICT_PATH, "r", encoding="utf-8") as f:
        cld = json.load(f)

    char_type_freq = Counter()
    for word, info in cld.items():
        for ch in info["characters"]:
            char_type_freq[ch["char"]] += 1

    df = pd.DataFrame(
        [(char, freq) for char, freq in char_type_freq.most_common()],
        columns=["character", "type_frequency"],
    )
    df["log_type_frequency"] = np.log(df["type_frequency"])

    os.makedirs(os.path.dirname(str(OUTPUT_PATH)), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"  Total words: {len(cld)}, unique characters: {len(df)}")
    print(f"  Type frequency range: {df['type_frequency'].min()} - {df['type_frequency'].max()}")
    print(f"  Saved -> {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
