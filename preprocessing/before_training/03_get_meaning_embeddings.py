"""
03_get_meaning_embeddings.py
============================
Extract contextual meaning embeddings for each character using Chinese RoBERTa.

Each character's embedding is extracted in the context of its parent word,
so the same character (e.g., "天") gets different embeddings in "天气" vs "天空".
This captures word-level semantic context.

Model: hfl/chinese-roberta-wwm-ext (768-dim hidden states)
Method: For word "自然", tokenize → [CLS] 自 然 [SEP]
        Character at position i → take hidden state at index i+1 (skip [CLS])

Input:  word_centric_dict.json
Output: word_meaning_embeddings.pt

Output structure (same as input, with 'meaning_embedding' tensor added):
{
    "自然": {
        "frequency": 115.37,
        "characters": [
            {"char": "自", "pinyin": "zi4", "position": 0, "meaning_embedding": tensor(768)},
            {"char": "然", "pinyin": "ran2", "position": 1, "meaning_embedding": tensor(768)}
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
OUTPUT_FILE = "data/embeddings/word_meaning_embeddings.pt"

# Model
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Load resources
# ==========================================
print(f"Using device: {DEVICE}")

print("Step 1: Loading dictionary...")
with open(DICT_FILE, "r", encoding="utf-8") as f:
    word_dict = json.load(f)
print(f"  Loaded {len(word_dict)} words.")

print("Step 2: Loading Chinese RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ==========================================
# 2. Extract meaning embeddings
# ==========================================
print("Step 3: Extracting meaning embeddings...")
final_data = {}
n_warnings = 0

for word, data in tqdm(word_dict.items(), total=len(word_dict), desc="Extracting"):
    # Tokenize the word (adds [CLS] and [SEP])
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        # Shape: [1, seq_len, 768] → [seq_len, 768]
        hidden_states = outputs.last_hidden_state.squeeze(0)

    # Extract embedding for each character
    new_char_list = []
    for char_info in data["characters"]:
        # BERT adds [CLS] at index 0, so character at position i → index i+1
        token_idx = char_info["position"] + 1

        if token_idx < len(hidden_states):
            char_entry = char_info.copy()
            char_entry["meaning_embedding"] = hidden_states[token_idx].cpu()
            new_char_list.append(char_entry)
        else:
            n_warnings += 1
            if n_warnings <= 10:
                print(f"  Warning: index out of bounds for '{word}' "
                      f"(pos={char_info['position']}, seq_len={len(hidden_states)})")

    final_data[word] = {
        "frequency": data["frequency"],
        "characters": new_char_list,
    }

# ==========================================
# 3. Save
# ==========================================
print(f"Step 4: Saving → {OUTPUT_FILE}")
torch.save(final_data, OUTPUT_FILE)

if n_warnings > 0:
    print(f"  Total warnings: {n_warnings}")
print(f"  Total words saved: {len(final_data)}")
print("✅ Done!")
