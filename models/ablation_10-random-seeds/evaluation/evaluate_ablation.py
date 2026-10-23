"""
Evaluate an ablation model (meaning-only or sound-only) and write a per-sample
CSV (entropy, ranks, logits) for the reported cell.

Mirrors evaluate_twolayers.py exactly (same per-sample inference loop, same
scale_K pinyin normalization, same model.eval(), same output columns). The ONLY
differences from the full-model eval are:
    - INPUT_DIM = 768            (single stream, not 1536)
    - OrthoNet takes model_type and the forward picks one stream
    - checkpoint / output paths carry {model_type} and use the ablation dir name
    - EPOCHS_TO_EVAL defaults to [11] (the reported cell), not range(1, 26)

The output CSV column schema is IDENTICAL to the full-model eval, so the existing
downstream *_trials_with_entropy.csv step works unchanged once repointed here.
model_type is carried by the filename: evaluation_CLD_<type>_hidden<H>_seed<S>.csv

Usage:
    python evaluate_ablation.py --model_type meaning --seed 42
    python evaluate_ablation.py --model_type sound   --seed 42 --hidden_dim 22
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", choices=["meaning", "sound"], required=True)
parser.add_argument("--hidden_dim", type=int, default=22)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_TYPE = args.model_type
HIDDEN_DIM = args.hidden_dim
SEED = args.seed

INPUT_DIM = 768                 # single stream; both ablations are 768-dim
TRAIN_EPOCHS = 11               # total epochs trained -> baked into the dir name
EPOCHS_TO_EVAL = [11]           # reported cell only. For the trajectory: range(1, 12)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_DICT_FILE = str(PROJECT_ROOT / "data/processed/word_centric_dict.json")
MEANING_FILE = str(PROJECT_ROOT / "data/embeddings/word_meaning_embeddings.pt")
PINYIN_FILE = str(PROJECT_ROOT / "data/embeddings/word_pinyin_embeddings.pt")

# dir is named with the TOTAL epoch count (TRAIN_EPOCHS); filename uses the per-epoch value
MODEL_ROOT = PROJECT_ROOT / "models/ablation_10-random-seeds"
CHECKPOINT_DIR = MODEL_ROOT / "weights" / f"CLD_{MODEL_TYPE}_ep{TRAIN_EPOCHS}_min1_lr4_hidden{HIDDEN_DIM}_normalized_seed{SEED}"
CHECKPOINT_TEMPLATE = f"CLD_{MODEL_TYPE}_ep{{epoch}}_min1_hidden{HIDDEN_DIM}_seed{SEED}.pth"
OUTPUT_CSV = MODEL_ROOT / "checkpoint" / f"evaluation_CLD_{MODEL_TYPE}_hidden{HIDDEN_DIM}_seed{SEED}.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. MODEL and DATASET DEFINITIONS
# ==========================================
class OrthoNet(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.model_type = model_type
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, meaning, pinyin):
        if self.model_type == "meaning":
            x = meaning
        elif self.model_type == "sound":
            x = pinyin
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)


class CharacterEvalDataset(Dataset):
    def __init__(self, json_file, meaning_file, pinyin_file):
        print(f"   Loading dictionary from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)

        print(f"   Loading embeddings...")
        self.meaning_data = torch.load(meaning_file, map_location='cpu', weights_only=False)
        self.pinyin_data = torch.load(pinyin_file, map_location='cpu', weights_only=False)

        # --- scale pinyin so its mean norm matches meaning's (same as full model & train) ---
        meaning_norms, pinyin_norms = [], []
        for word in self.meaning_data:
            if word not in self.pinyin_data:
                continue
            for c in self.meaning_data[word]["characters"]:
                meaning_norms.append(torch.norm(c["meaning_embedding"]).item())
            for c in self.pinyin_data[word]["characters"]:
                pinyin_norms.append(torch.norm(c["pinyin_embedding"]).item())

        self.scale_K = np.mean(meaning_norms) / np.mean(pinyin_norms)
        print(f"   Scale factor K: {self.scale_K:.4f}")
        for word in self.pinyin_data:
            for c in self.pinyin_data[word]["characters"]:
                c["pinyin_embedding"] = c["pinyin_embedding"] * self.scale_K

        self.sample_list = []
        self.unique_chars = set()

        for word, info in self.meta_data.items():
            if word not in self.meaning_data or word not in self.pinyin_data:
                continue
            raw_freq = info.get('frequency', 0.0)
            for ci in info['characters']:
                self.sample_list.append({
                    'parent_word': word,
                    'char_text': ci['char'],
                    'vec_index': ci['position'],
                    'word_freq': raw_freq
                })
                self.unique_chars.add(ci['char'])

        self.char_to_id = {c: i for i, c in enumerate(sorted(self.unique_chars))}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        print(f"   Samples: {len(self.sample_list)}, Classes: {len(self.char_to_id)}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        s = self.sample_list[idx]
        m = self.meaning_data[s['parent_word']]['characters'][s['vec_index']]['meaning_embedding']
        p = self.pinyin_data[s['parent_word']]['characters'][s['vec_index']]['pinyin_embedding']
        label_id = self.char_to_id[s['char_text']]
        return m.float(), p.float(), torch.tensor(label_id, dtype=torch.long), idx


def get_rank_of_target(logits, target_id):
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
    return rank


# ==========================================
# 3. EVALUATION PIPELINE
# ==========================================
def evaluate_single_epoch(model, dataset, epoch, checkpoint_path):
    print(f"\n--- [Epoch {epoch}] Evaluation Start ---")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return []

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Ep{epoch} Infer"):
            m_vec, p_vec, label, idx = dataset[i]
            m_vec = m_vec.unsqueeze(0).to(device)
            p_vec = p_vec.unsqueeze(0).to(device)
            target_id = label.item()

            logits = model(m_vec, p_vec)
            probs = torch.softmax(logits, dim=1)

            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=1).item()

            topk_probs, topk_indices = torch.topk(probs, k=2, dim=1)
            topk_logits, _ = torch.topk(logits, k=2, dim=1)

            top1_id = topk_indices[0][0].item()
            top2_id = topk_indices[0][1].item()
            is_correct = (top1_id == target_id)
            target_rank = 1 if is_correct else get_rank_of_target(logits[0], target_id)

            sample_info = dataset.sample_list[idx]

            results.append({
                "Epoch": epoch,
                "Context Word": sample_info['parent_word'],
                "Target Char": dataset.id_to_char[target_id],
                "Frequency": sample_info['word_freq'],
                "Is Correct": is_correct,
                "Target Rank": target_rank,
                "Top1 Char": dataset.id_to_char[top1_id],
                "Top2 Char": dataset.id_to_char[top2_id],
                "Entropy": round(entropy, 4),
                "Top1 Logit": round(topk_logits[0][0].item(), 4),
                "Top2 Logit": round(topk_logits[0][1].item(), 4),
                "Logit Gap": round(topk_logits[0][0].item() - topk_logits[0][1].item(), 4),
                "Top1 Prob": round(topk_probs[0][0].item(), 4),
                "Top2 Prob": round(topk_probs[0][1].item(), 4),
                "Prob Gap": round(topk_probs[0][0].item() - topk_probs[0][1].item(), 4),
            })

    return results


# ==========================================
# 4. MAIN
# ==========================================
def main():
    print(f"Evaluating ablation: model_type={MODEL_TYPE}, hidden_dim={HIDDEN_DIM}, seed={SEED}")
    print(f"Input dim: {INPUT_DIM} | epochs to eval: {list(EPOCHS_TO_EVAL)}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    required = [CHECKPOINT_DIR / CHECKPOINT_TEMPLATE.format(epoch=epoch) for epoch in EPOCHS_TO_EVAL]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained ablation weights:\n" + "\n".join(missing))
    dataset = CharacterEvalDataset(JSON_DICT_FILE, MEANING_FILE, PINYIN_FILE)
    output_dim = len(dataset.char_to_id)

    model = OrthoNet(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, output_dim).to(device)

    all_results = []
    for epoch in EPOCHS_TO_EVAL:
        ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_TEMPLATE.format(epoch=epoch))
        epoch_results = evaluate_single_epoch(model, dataset, epoch, ckpt_path)
        all_results.extend(epoch_results)
        print(f"Epoch {epoch} done. Added {len(epoch_results)} rows.")

    if not all_results:
        raise SystemExit("No results produced. Check that the checkpoints exist at the paths above.")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved: {OUTPUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
