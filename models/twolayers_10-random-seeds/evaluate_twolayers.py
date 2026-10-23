"""
Evaluate two-hidden-layer model, retaining all diagnostics plus target logits.

Reads the original embeddings/dictionary and existing .pth checkpoints in this project's weights directory.
Writes the original evaluation CSV schema with Target Logit beside Target Rank.
No training is performed. Dependencies must already exist in the environment.

Usage:
    python evaluate_twolayers.py --hidden_dim 22 --seed 42
    python evaluate_twolayers.py --hidden_dim 22 --seed 42 --epochs 11 --output_csv pilot_seed42.csv

All ten seeds use the same model directory.
Default: epochs 1-25. Use --model_root to override the model directory.
"""

import json
import os
import argparse
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually in the evaluation environment.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_DIR / "models/twolayers_10-random-seeds"
DEFAULT_EPOCHS = list(range(1, 26))
INPUT_DIM = 1536
LOGIT_DECIMALS = 4
CHECKPOINT_SCALE_RTOL = 1e-7
CHECKPOINT_SCALE_ATOL = 1e-10

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--model_root", type=Path, help="Folder containing checkpoint directories and evaluation/.")
parser.add_argument("--epochs", type=int, nargs="+", help="Evaluate only these epochs; requires --output_csv.")
parser.add_argument("--output_csv", type=Path, help="Override the output file, e.g. for a pilot evaluation.")
args = parser.parse_args()
if args.epochs is not None and args.output_csv is None:
    parser.error("--epochs requires --output_csv so a partial run cannot replace the complete evaluation CSV.")
if args.epochs is not None and (len(set(args.epochs)) != len(args.epochs) or
                                any(epoch not in DEFAULT_EPOCHS for epoch in args.epochs)):
    parser.error("--epochs must contain unique values from 1 to 25.")
HIDDEN_DIM = args.hidden_dim
SEED = args.seed
EPOCHS_TO_EVAL = sorted(args.epochs) if args.epochs is not None else DEFAULT_EPOCHS

# All seeds share one checkpoint collection; retain a command-line path override.
MODEL_ROOT = args.model_root if args.model_root is not None else DEFAULT_MODEL_ROOT

JSON_DICT_FILE = PROJECT_DIR / "data/processed/word_centric_dict.json"
MEANING_FILE = PROJECT_DIR / "data/embeddings/word_meaning_embeddings.pt"
PINYIN_FILE = PROJECT_DIR / "data/embeddings/word_pinyin_embeddings.pt"

CHECKPOINT_DIR = MODEL_ROOT / "weights" / f"CLD_twolayers_ep25_min1_lr4_hidden{HIDDEN_DIM}_normalized_seed{SEED}"
CHECKPOINT_TEMPLATE = f"CLD_twolayers_ep{{epoch}}_min1_hidden{HIDDEN_DIM}_seed{SEED}.pth"
OUTPUT_CSV = args.output_csv if args.output_csv is not None else MODEL_ROOT / "checkpoint" / f"evaluation_CLD_twolayers_hidden{HIDDEN_DIM}_seed{SEED}.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. MODEL and DATASET DEFINITIONS
# ==========================================
class OrthoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, meaning, pinyin):
        x = torch.cat([meaning, pinyin], dim=1)
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Identity and input scaling must match the weights; do not silently relabel.
    if "char_to_id" in checkpoint and checkpoint["char_to_id"] != dataset.char_to_id:
        raise ValueError(f"Character index mismatch: {checkpoint_path}")
    if "scale_K" in checkpoint and not np.isclose(
            checkpoint["scale_K"], dataset.scale_K,
            rtol=CHECKPOINT_SCALE_RTOL, atol=CHECKPOINT_SCALE_ATOL):
        raise ValueError(f"Pinyin scaling mismatch: {checkpoint_path}")
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
                "Target Logit": round(logits[0, target_id].item(), LOGIT_DECIMALS),
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
    print(f"Evaluating: hidden_dim={HIDDEN_DIM}, seed={SEED}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")

    # Require all requested inputs before loading embeddings or replacing output.
    paths = [JSON_DICT_FILE, MEANING_FILE, PINYIN_FILE] + [
        CHECKPOINT_DIR / CHECKPOINT_TEMPLATE.format(epoch=epoch) for epoch in EPOCHS_TO_EVAL
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluation inputs:\n" + "\n".join(missing))

    dataset = CharacterEvalDataset(JSON_DICT_FILE, MEANING_FILE, PINYIN_FILE)
    output_dim = len(dataset.char_to_id)

    model = OrthoNet(INPUT_DIM, HIDDEN_DIM, output_dim).to(device)

    all_results = []
    for epoch in EPOCHS_TO_EVAL:
        ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_TEMPLATE.format(epoch=epoch))
        epoch_results = evaluate_single_epoch(model, dataset, epoch, ckpt_path)
        all_results.extend(epoch_results)
        print(f"Epoch {epoch} done. Added {len(epoch_results)} rows.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    # Replace only after the complete CSV is successfully written.
    temporary_csv = OUTPUT_CSV.with_suffix(OUTPUT_CSV.suffix + ".tmp")
    df.to_csv(temporary_csv, index=False, encoding='utf-8-sig')
    os.replace(temporary_csv, OUTPUT_CSV)
    print(f"Saved: {OUTPUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
