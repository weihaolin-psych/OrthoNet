"""
04_get_pinyin_embeddings.py
===========================
Extract pinyin embeddings for each character using ChineseBERT's pinyin layer.

Unlike meaning embeddings, pinyin embeddings are NOT contextual — each character's
pinyin string (e.g., "zi4") always maps to the same embedding regardless of word context.
We use only the pinyin embedding layer from ChineseBERT (not the full model).

Model: iioSnail/ChineseBERT-base (pinyin layer only, 768-dim output)
Method: Convert pinyin string to fixed-length (8) ID sequence → pass through pinyin layer

Input:  word_centric_dict.json
Output: word_pinyin_embeddings.pt

Output structure (same as input, with 'pinyin_embedding' tensor added):
{
    "自然": {
        "frequency": 115.37,
        "characters": [
            {"char": "自", "pinyin": "zi4", "position": 0, "pinyin_embedding": tensor(768)},
            {"char": "然", "pinyin": "ran2", "position": 1, "pinyin_embedding": tensor(768)}
        ]
    }
}
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# CONFIGURATION — adjust paths here
# ==========================================
# Input
DICT_FILE = "data/processed/word_centric_dict.json"

# Output
OUTPUT_FILE = "data/embeddings/word_pinyin_embeddings.pt"

# Model
MODEL_NAME = "iioSnail/ChineseBERT-base"

# ChineseBERT pinyin encoding: fixed length
PINYIN_MAX_LEN = 8

# ==========================================
# 1. Load resources
# ==========================================
print("Step 1: Loading dictionary...")
with open(DICT_FILE, "r", encoding="utf-8") as f:
    word_dict = json.load(f)
print(f"  Loaded {len(word_dict)} words.")

print("Step 2: Loading ChineseBERT pinyin layer...")
# use_fast=False is required to access pinyin_dict
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()

# Extract only the pinyin embedding layer (no need to run full BERT)
pinyin_layer = model.bert.embeddings.pinyin_embeddings

# ==========================================
# 2. Helper function
# ==========================================
def encode_pinyin(pinyin_str):
    """
    Convert a pinyin string (e.g., 'zi4') to a 768-dim embedding
    using ChineseBERT's pinyin layer.

    Returns: tensor(768) or None if encoding fails.
    """
    if not pinyin_str:
        return None

    # Encode pinyin characters to IDs (fixed length = 8)
    ids = [0] * PINYIN_MAX_LEN
    try:
        char2idx = tokenizer.pinyin_dict["char2idx"]
        for i, ch in enumerate(pinyin_str[:PINYIN_MAX_LEN]):
            if ch in char2idx:
                ids[i] = char2idx[ch]
    except (KeyError, AttributeError):
        return None

    # Shape: [batch=1, seq=1, pinyin_len=8]
    tensor_in = torch.tensor([ids], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = pinyin_layer(tensor_in).squeeze()  # → tensor(768)

    return out

# ==========================================
# 3. Extract pinyin embeddings
# ==========================================
print("Step 3: Extracting pinyin embeddings...")
final_data = {}
n_missing = 0

for word, data in tqdm(word_dict.items(), total=len(word_dict), desc="Extracting"):
    new_char_list = []

    for char_info in data["characters"]:
        pinyin_str = char_info.get("pinyin", "")
        vec = encode_pinyin(pinyin_str)

        char_entry = char_info.copy()
        if vec is not None:
            char_entry["pinyin_embedding"] = vec
        else:
            n_missing += 1
            if n_missing <= 10:
                print(f"  Warning: no pinyin embedding for '{char_info['char']}' "
                      f"(pinyin='{pinyin_str}') in word '{word}'")

        new_char_list.append(char_entry)

    final_data[word] = {
        "frequency": data["frequency"],
        "characters": new_char_list,
    }

# ==========================================
# 4. Save
# ==========================================
print(f"Step 4: Saving → {OUTPUT_FILE}")
torch.save(final_data, OUTPUT_FILE)

if n_missing > 0:
    print(f"  Characters with missing pinyin embedding: {n_missing}")
print(f"  Total words saved: {len(final_data)}")
print("✅ Done!")
