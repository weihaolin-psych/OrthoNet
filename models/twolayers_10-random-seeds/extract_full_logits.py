"""Export complete pre-softmax candidate logits from an existing checkpoint.

Run from the project root: python -u extract_full_logits.py --seed 42
Reads the original dictionary, embeddings, checkpoint and evaluation CSV.
Writes one uncompressed, indexed NPZ; never trains or replaces an existing file.
Requires NumPy, PyTorch and tqdm; missing packages must be installed manually.
"""

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}. Install it manually in the extraction environment.")

# ---- Configuration ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_DIR / "models/twolayers_10-random-seeds"
DEFAULT_HIDDEN_DIM = 22
DEFAULT_EPOCH = 11
INPUT_DIM = 1536
EXPECTED_ROWS = 106_844
EXPECTED_CANDIDATES = 4_677
LOGIT_DECIMALS = 4
CHECKPOINT_SCALE_RTOL = 1e-7
CHECKPOINT_SCALE_ATOL = 1e-10
VERIFY_CHUNK_ROWS = 1024
DISK_RESERVE_BYTES = 256 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


# ---- Model and dataset definitions -----------------------------------------
# These definitions mirror the original evaluator; importing that executable
# would parse its arguments and couple the two independent workflows.
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
        with open(json_file, encoding="utf-8") as stream:
            self.meta_data = json.load(stream)
        self.meaning_data = torch.load(meaning_file, map_location="cpu", weights_only=False)
        self.pinyin_data = torch.load(pinyin_file, map_location="cpu", weights_only=False)

        # Preserve the original norm order and ratio, before modifying pinyin.
        meaning_norms, pinyin_norms = [], []
        for word in self.meaning_data:
            if word not in self.pinyin_data:
                continue
            for char in self.meaning_data[word]["characters"]:
                meaning_norms.append(torch.norm(char["meaning_embedding"]).item())
            for char in self.pinyin_data[word]["characters"]:
                pinyin_norms.append(torch.norm(char["pinyin_embedding"]).item())
        require(bool(meaning_norms) and bool(pinyin_norms), "Empty embedding inventory.")
        require(np.mean(pinyin_norms) > 0, "Pinyin norm must be positive.")
        self.scale_K = np.mean(meaning_norms) / np.mean(pinyin_norms)
        require(np.isfinite(self.scale_K) and self.scale_K > 0, "Invalid input scale_K.")
        for word in self.pinyin_data:
            for char in self.pinyin_data[word]["characters"]:
                char["pinyin_embedding"] = char["pinyin_embedding"] * self.scale_K

        # Dictionary insertion order and every original position are retained.
        self.sample_list = []
        unique_chars = set()
        for word, info in self.meta_data.items():
            if word not in self.meaning_data or word not in self.pinyin_data:
                continue
            for char in info["characters"]:
                position = char["position"]
                require(isinstance(position, int) and position >= 0,
                        f"Invalid position: {word}, {position}")
                require(position < len(self.meaning_data[word]["characters"]) and
                        position < len(self.pinyin_data[word]["characters"]),
                        f"Embedding position missing: {word}, {position}")
                self.sample_list.append({"parent_word": word, "char_text": char["char"],
                                         "vec_index": position})
                unique_chars.add(char["char"])
        self.char_to_id = {char: i for i, char in enumerate(sorted(unique_chars))}

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.sample_list[index]
        word, position = sample["parent_word"], sample["vec_index"]
        meaning = self.meaning_data[word]["characters"][position]["meaning_embedding"]
        pinyin = self.pinyin_data[word]["characters"][position]["pinyin_embedding"]
        return meaning.float(), pinyin.float(), self.char_to_id[sample["char_text"]]


# ---- Row/column identity and original CSV comparison ------------------------
def make_index_arrays(dataset, hidden_dim, epoch, seed):
    samples = dataset.sample_list
    return {
        "candidate_chars": np.asarray(list(dataset.char_to_id), dtype=str),
        "context_words": np.asarray([s["parent_word"] for s in samples], dtype=str),
        "target_chars": np.asarray([s["char_text"] for s in samples], dtype=str),
        "positions": np.asarray([s["vec_index"] for s in samples], dtype=np.int64),
        "target_ids": np.asarray([dataset.char_to_id[s["char_text"]] for s in samples], dtype=np.int64),
        "hidden_dim": np.asarray(hidden_dim, dtype=np.int64),
        "epoch": np.asarray(epoch, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
    }


def read_reference(path, epoch, indices):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"Epoch", "Context Word", "Target Char", "Target Logit", "Top1 Logit", "Top2 Logit"}
        require(needed.issubset(reader.fieldnames or []), f"Missing reference columns: {path}")
        rows = [row for row in reader if int(row["Epoch"]) == epoch]
    require(len(rows) == len(indices["target_ids"]), "Reference CSV row count mismatch.")
    for i, row in enumerate(rows):
        require(row["Context Word"] == indices["context_words"][i] and
                row["Target Char"] == indices["target_chars"][i],
                f"Reference CSV input order mismatch at row {i}.")
    reference = np.asarray([[float(row[key]) for key in
                            ("Target Logit", "Top1 Logit", "Top2 Logit")] for row in rows])
    require(np.isfinite(reference).all(), "Nonfinite reference logits.")
    return reference


def compare_reference(logits, target_id, reference_row, row_index):
    # Only the top values matter here, so tied candidate identities are harmless.
    top = torch.topk(logits, k=2).values
    values = [logits[target_id].item(), top[0].item(), top[1].item()]
    rounded = [round(value, LOGIT_DECIMALS) for value in values]
    require(np.array_equal(rounded, reference_row),
            f"Original CSV logit mismatch at row {row_index}: {rounded} vs {reference_row.tolist()}")


# ---- Archive writing and exact readback ------------------------------------
def write_and_verify_archive(path, matrix, indices):
    # np.savez uses uncompressed ZIP storage and keeps the matrix's float32 dtype.
    np.savez(path, logits=matrix, **indices)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"logits", *indices}, "Archive fields differ.")
        for key, expected in indices.items():
            require(np.array_equal(archive[key], expected), f"Archive index mismatch: {key}")
        restored = archive["logits"]
        require(restored.dtype == np.float32 and restored.shape == matrix.shape,
                "Archive matrix shape/dtype mismatch.")
        require(np.array_equal(indices["candidate_chars"][indices["target_ids"]],
                               indices["target_chars"]), "Target column mapping mismatch.")
        for start in range(0, len(matrix), VERIFY_CHUNK_ROWS):
            stop = start + VERIFY_CHUNK_ROWS
            require(np.isfinite(restored[start:stop]).all(), "Nonfinite saved logits.")
            require(np.array_equal(restored[start:stop], matrix[start:stop]),
                    f"Archive numerical readback mismatch at row {start}.")


def export_checkpoint(args, temporary_dir):
    model_root = args.model_root
    checkpoint_path = model_root / "weights" / (
        f"CLD_twolayers_ep25_min1_lr4_hidden{args.hidden_dim}_normalized_seed{args.seed}"
    ) / f"CLD_twolayers_ep{args.epoch}_min1_hidden{args.hidden_dim}_seed{args.seed}.pth"
    reference_path = (args.reference_csv if args.reference_csv is not None else
                      model_root / "checkpoint" / f"evaluation_CLD_twolayers_hidden{args.hidden_dim}_seed{args.seed}.csv")
    inputs = [args.project_dir / name for name in
              ("data/processed/word_centric_dict.json", "data/embeddings/word_meaning_embeddings.pt", "data/embeddings/word_pinyin_embeddings.pt")]
    missing = [str(path) for path in [*inputs, checkpoint_path, reference_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing extraction inputs:\n" + "\n".join(missing))

    dataset = CharacterEvalDataset(*inputs)
    shape = (len(dataset), len(dataset.char_to_id))
    require(shape == (EXPECTED_ROWS, EXPECTED_CANDIDATES), f"Unexpected CLD inventory: {shape}")
    indices = make_index_arrays(dataset, args.hidden_dim, args.epoch, args.seed)
    reference = read_reference(reference_path, args.epoch, indices)
    required_bytes = 2 * shape[0] * shape[1] * np.dtype(np.float32).itemsize + DISK_RESERVE_BYTES
    require(shutil.disk_usage(temporary_dir).free >= required_bytes,
            f"Need at least {required_bytes / 1e9:.2f} GB free for matrix, archive and reserve.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting hidden{args.hidden_dim}, epoch{args.epoch}, seed{args.seed}; {device}; shape={shape}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "char_to_id" in checkpoint:
        require(checkpoint["char_to_id"] == dataset.char_to_id, "Checkpoint character mapping mismatch.")
    else:
        print("Checkpoint has no char_to_id; using original sorted mapping and full CSV comparison.")
    if "scale_K" in checkpoint:
        require(np.isclose(checkpoint["scale_K"], dataset.scale_K,
                           rtol=CHECKPOINT_SCALE_RTOL, atol=CHECKPOINT_SCALE_ATOL),
                "Checkpoint pinyin scaling mismatch.")
    else:
        print("Checkpoint has no scale_K; using original scaling and full CSV comparison.")
    model = OrthoNet(INPUT_DIM, args.hidden_dim, shape[1]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    matrix = np.lib.format.open_memmap(temporary_dir / "logits.npy", mode="w+",
                                       dtype=np.float32, shape=shape)
    try:
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Full logits"):
                meaning, pinyin, target_id = dataset[i]
                logits = model(meaning.unsqueeze(0).to(device), pinyin.unsqueeze(0).to(device))[0]
                require(logits.dtype == torch.float32 and tuple(logits.shape) == (shape[1],),
                        f"Unexpected output dtype/shape at row {i}.")
                values = logits.cpu().numpy()
                require(np.isfinite(values).all(), f"Nonfinite output at row {i}.")
                compare_reference(logits, target_id, reference[i], i)
                matrix[i] = values
        matrix.flush()
        archive_path = temporary_dir / "complete.npz"
        write_and_verify_archive(archive_path, matrix, indices)
    finally:
        # Close the mapping before TemporaryDirectory removes its files (Windows too).
        matrix._mmap.close()
    return archive_path


# ---- Command line and exclusive publication --------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--project_dir", type=Path, default=PROJECT_DIR,
                        help="Project root containing data/processed and data/embeddings.")
    parser.add_argument("--model_root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output_dir", type=Path, help="Default: MODEL_ROOT/full_logit")
    parser.add_argument("--reference_csv", type=Path,
                        help="Default: existing evaluation CSV under MODEL_ROOT/checkpoint")
    args = parser.parse_args()
    if args.hidden_dim <= 0 or not 1 <= args.epoch <= 25 or args.seed < 0:
        parser.error("Require hidden_dim > 0, epoch in 1..25 and seed >= 0.")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.model_root / "full_logit"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"logits_hidden{args.hidden_dim}_epoch{args.epoch}_seed{args.seed}.npz"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_path}")
    lock_path = final_path.with_suffix(".lock")
    # A second job for the same checkpoint must not run concurrently.
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(lock_fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        with tempfile.TemporaryDirectory(prefix=f".{final_path.stem}_", dir=output_dir) as directory:
            archive_path = export_checkpoint(args, Path(directory))
            # Windows rename refuses replacement; on Linux use a hard link for
            # the same no-overwrite guarantee (ordinary Unix rename replaces).
            if os.name == "nt":
                os.rename(archive_path, final_path)
            else:
                os.link(archive_path, final_path)
            print(f"Verified and saved: {final_path} ({final_path.stat().st_size / 1e9:.3f} GB)")
    finally:
        lock_path.unlink()


if __name__ == "__main__":
    main()
