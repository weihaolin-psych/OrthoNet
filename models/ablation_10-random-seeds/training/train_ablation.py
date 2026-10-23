"""
Train ablation models (meaning-only or sound-only) at the reported cell.

Mirrors train_twolayers.py exactly (same OrthoNet body, same scale_K pinyin
normalization, same determinism, same training-order logic). The ONLY thing
that changes is which input stream feeds layer1:
    meaning : meaning             -> input_dim 768
    sound   : pinyin (normalized) -> input_dim 768

OUTPUT (per run = one model_type + one seed):
    <OUTPUT_ROOT>/<output dir>/
        char_to_id.json                      # character -> id map
        CLD_<type>_ep1_..._seed<S>.pth        # one checkpoint per epoch
        ...                                   # (you use the ep11 checkpoint)
        CLD_<type>_ep<EPOCHS>_..._seed<S>.pth

Usage:
    python train_ablation.py --model_type meaning --seed 42
    python train_ablation.py --model_type sound   --seed 42 --hidden_dim 22 --epochs 11
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os, random, json, math, argparse
from tqdm import tqdm

# ============================================================
# CONFIGURATION  (edit here -- no need to read the rest)
# ============================================================
# --- input data files ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_DICT_FILE = str(PROJECT_ROOT / "data/processed/word_centric_dict.json")
MEANING_FILE = str(PROJECT_ROOT / "data/embeddings/word_meaning_embeddings.pt")
PINYIN_FILE = str(PROJECT_ROOT / "data/embeddings/word_pinyin_embeddings.pt")

# --- where outputs go ---
OUTPUT_ROOT = str(PROJECT_ROOT / "models/ablation_10-random-seeds/weights")
# how each run's folder + checkpoint files are named (placeholders filled at runtime)
OUTPUT_DIR_TEMPLATE = "CLD_{model_type}_ep{epochs}_min{min_occ}_lr4_hidden{hidden_dim}_normalized_seed{seed}"
CHECKPOINT_TEMPLATE = "CLD_{model_type}_ep{epoch}_min{min_occ}_hidden{hidden_dim}_seed{seed}.pth"

# --- fixed training hyperparameters (same as the full model) ---
ALPHA           = 0.75      # frequency-weighting exponent
MIN_OCCURRENCES = 1         # min copies of the rarest sample per epoch
BATCH_SIZE      = 512
LEARNING_RATE   = 0.0001
INPUT_DIM       = 768       # single stream; both ablations are 768-dim
DROPOUT         = 0.3

# --- per-run defaults (overridable on the command line; the shell loop sets these) ---
DEFAULT_MODEL_TYPE = "meaning"   # "meaning" or "sound"
DEFAULT_HIDDEN_DIM = 22          # the reported cell
DEFAULT_EPOCHS     = 11          # the reported cell (saves ep1..ep11)
DEFAULT_SEED       = 42
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Model: same body as the full OrthoNet; input stream chosen by model_type
# ==========================================
class OrthoNet(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.model_type = model_type
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, meaning, pinyin):
        if self.model_type == "full":
            x = torch.cat([meaning, pinyin], dim=1)
        elif self.model_type == "meaning":
            x = meaning
        elif self.model_type == "sound":
            x = pinyin
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.relu(self.layer2(x))
        return self.output_layer(x)


# ==========================================
# Dataset: identical to train_twolayers.py (pinyin normalized by scale_K)
# ==========================================
class CharacterTrainingDataset(Dataset):
    def __init__(self, json_file, meaning_file, pinyin_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)
        self.meaning_data = torch.load(meaning_file, map_location='cpu', weights_only=False)
        self.pinyin_data = torch.load(pinyin_file, map_location='cpu', weights_only=False)

        # --- scale pinyin so its mean norm matches meaning's (same as full model) ---
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

        self.sample_list, self.sample_weights, self.unique_chars = [], [], set()
        for word, info in self.meta_data.items():
            if word not in self.meaning_data or word not in self.pinyin_data:
                continue
            smooth_freq = math.pow(info.get('frequency', 0.0001), ALPHA)
            for ci in info['characters']:
                self.sample_list.append({
                    'parent_word': word,
                    'char_text': ci['char'],
                    'vec_index': ci['position']
                })
                self.sample_weights.append(smooth_freq)
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
        return m.float(), p.float(), torch.tensor(self.char_to_id[s['char_text']], dtype=torch.long)


def generate_deterministic_indices(dataset):
    weights = np.array(dataset.sample_weights)
    scaling_factor = MIN_OCCURRENCES / np.min(weights)
    counts = np.ceil(weights * scaling_factor).astype(int)
    expanded = []
    for idx, count in enumerate(counts):
        expanded.extend([idx] * count)
    print(f"   Expanded samples per epoch: {len(expanded)}")
    random.shuffle(expanded)
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["meaning", "sound"], default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    MODEL_TYPE, HIDDEN_DIM, EPOCHS, SEED = args.model_type, args.hidden_dim, args.epochs, args.seed

    # build the output dir from the template in CONFIGURATION
    output_path = os.path.join(OUTPUT_ROOT, OUTPUT_DIR_TEMPLATE.format(
        model_type=MODEL_TYPE, epochs=EPOCHS, min_occ=int(MIN_OCCURRENCES),
        hidden_dim=HIDDEN_DIM, seed=SEED))

    # --- determinism block (matches train_twolayers.py) ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(output_path, exist_ok=True)
    print(f"Model type: {MODEL_TYPE} | Input dim: {INPUT_DIM} | hidden_dim: {HIDDEN_DIM} | "
          f"epochs: {EPOCHS} | seed: {SEED} | device: {device}")
    print(f"Output dir: {output_path}\n")

    dataset = CharacterTrainingDataset(JSON_DICT_FILE, MEANING_FILE, PINYIN_FILE)
    with open(os.path.join(output_path, "char_to_id.json"), "w", encoding='utf-8') as f:
        json.dump(dataset.char_to_id, f, ensure_ascii=False, indent=4)

    expanded_indices = generate_deterministic_indices(dataset)

    model = OrthoNet(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, len(dataset.char_to_id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print(f"\n--- Training {MODEL_TYPE}: hidden_dim={HIDDEN_DIM}, epochs=1-{EPOCHS}, seed={SEED} ---")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        random.shuffle(expanded_indices)
        num_batches = math.ceil(len(expanded_indices) / BATCH_SIZE)

        for i in tqdm(range(0, len(expanded_indices), BATCH_SIZE),
                      desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", mininterval=30):
            batch_idx = expanded_indices[i:i+BATCH_SIZE]
            bm, bp, bl = zip(*[dataset[j] for j in batch_idx])
            tm = torch.stack(bm).to(device)
            tp = torch.stack(bp).to(device)
            tl = torch.stack(bl).to(device)

            optimizer.zero_grad()
            loss = criterion(model(tm, tp), tl)     # forward picks the stream
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/num_batches:.4f}")

        ckpt_name = CHECKPOINT_TEMPLATE.format(
            model_type=MODEL_TYPE, epoch=epoch+1,
            min_occ=int(MIN_OCCURRENCES), hidden_dim=HIDDEN_DIM, seed=SEED)
        torch.save({
            'model_state_dict': model.state_dict(),
            'char_to_id': dataset.char_to_id,
            'scale_K': dataset.scale_K,
            'config': {
                'model_type': MODEL_TYPE, 'input_dim': INPUT_DIM,
                'hidden_dim': HIDDEN_DIM, 'learning_rate': LEARNING_RATE,
                'alpha': ALPHA, 'min_occurrences': MIN_OCCURRENCES,
                'batch_size': BATCH_SIZE, 'seed': SEED, 'normalized': True
            },
        }, os.path.join(output_path, ckpt_name))
        print(f"Saved: {ckpt_name}")


if __name__ == "__main__":
    main()
