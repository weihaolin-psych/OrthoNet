"""
Train two-hidden-layer model with parameterized hidden_dim and seed.

Usage:
    python train_twolayers.py --hidden_dim 22 --epochs 25 --seed 42
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os, random, json, math, argparse
from tqdm import tqdm

# ---- Configuration: command line and training settings --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, required=True)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

ALPHA = 0.75
MIN_OCCURRENCES = 1
BATCH_SIZE = 512
EPOCHS = args.epochs
HIDDEN_DIM = args.hidden_dim
SEED = args.seed
LEARNING_RATE = 0.0001

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JSON_DICT_FILE = str(PROJECT_ROOT / "data/processed/word_centric_dict.json")
MEANING_FILE = str(PROJECT_ROOT / "data/embeddings/word_meaning_embeddings.pt")
PINYIN_FILE = str(PROJECT_ROOT / "data/embeddings/word_pinyin_embeddings.pt")
MODEL_ROOT = PROJECT_ROOT / "models/twolayers_10-random-seeds"
OUTPUT_PATH = str(MODEL_ROOT / "weights" / f"CLD_twolayers_ep{EPOCHS}_min{int(MIN_OCCURRENCES)}_lr4_hidden{HIDDEN_DIM}_normalized_seed{SEED}")
CHECKPOINT_TEMPLATE = "CLD_twolayers_ep{epoch}_min{min_occ}_hidden{hidden_dim}_seed{seed}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class CharacterTrainingDataset(Dataset):
    def __init__(self, json_file, meaning_file, pinyin_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    dataset = CharacterTrainingDataset(JSON_DICT_FILE, MEANING_FILE, PINYIN_FILE)
    with open(os.path.join(OUTPUT_PATH, "char_to_id.json"), "w", encoding='utf-8') as f:
        json.dump(dataset.char_to_id, f, ensure_ascii=False, indent=4)

    expanded_indices = generate_deterministic_indices(dataset)

    model = OrthoNet(1536, HIDDEN_DIM, len(dataset.char_to_id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    print(f"\n--- Training: hidden_dim={HIDDEN_DIM}, epochs=1-{EPOCHS}, seed={SEED} ---")
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
            loss = criterion(model(tm, tp), tl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/num_batches:.4f}")

        ckpt_name = CHECKPOINT_TEMPLATE.format(
            epoch=epoch+1, min_occ=1, hidden_dim=HIDDEN_DIM, seed=SEED
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'char_to_id': dataset.char_to_id,
            'scale_K': dataset.scale_K,
            'config': {
                'hidden_dim': HIDDEN_DIM,
                'learning_rate': LEARNING_RATE,
                'alpha': ALPHA,
                'min_occurrences': MIN_OCCURRENCES,
                'batch_size': BATCH_SIZE,
                'seed': SEED,
                'normalized': True
            },
        }, os.path.join(OUTPUT_PATH, ckpt_name))
        print(f"Saved: {ckpt_name}")


if __name__ == "__main__":
    main()
