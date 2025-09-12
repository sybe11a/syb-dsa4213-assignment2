"""
Build Subword Vocab + Re-encode splits (SentencePiece)
------------------------------------------------------
- Trains a SentencePiece model (BPE or unigram) on the TRAIN split
- Saves:
    data/subword.model     (SentencePiece model file)
    data/subword.vocab     (human-readable vocab)
    data/subword.json      (token->id dict, for PyTorch)
- Re-encodes all splits with this model:
    data/split/train_subword.txt
    data/split/val_subword.txt
    data/split/test_subword.txt
"""

from __future__ import annotations
from pathlib import Path
import json
import sentencepiece as spm

DATA_DIR = Path("data")
SPLIT_DIR = DATA_DIR / "split"

TRAIN_PATH = SPLIT_DIR / "train.txt"
VAL_PATH   = SPLIT_DIR / "val.txt"
TEST_PATH  = SPLIT_DIR / "test.txt"

# outputs
MODEL_PREFIX = str(DATA_DIR / "subword")
MODEL_FILE   = DATA_DIR / "subword.model"
VOCAB_FILE   = DATA_DIR / "subword.vocab"
VOCAB_JSON   = DATA_DIR / "subword.json"

# config
VOCAB_SIZE = 2000        # adjust depending on dataset size
MODEL_TYPE = "bpe"       # "bpe" or "unigram"

def run():
    # ------------------------
    # 1. Train SentencePiece
    # ------------------------
    print(f"[info] Training SentencePiece on {TRAIN_PATH} ...")
    spm.SentencePieceTrainer.Train(
        input=str(TRAIN_PATH),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=1.0,  # English text = 1.0
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
    )

    print(f"[done] Model written to {MODEL_FILE}, vocab to {VOCAB_FILE}")

    # ------------------------
    # 2. Save vocab as JSON
    # ------------------------
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_FILE))

    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    VOCAB_JSON.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Subword vocab saved to {VOCAB_JSON}")

    # ------------------------
    # 3. Re-encode splits
    # ------------------------
    for split, path in {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        encoded = [" ".join(sp.encode(line, out_type=str)) for line in lines]
        out_path = SPLIT_DIR / f"{split}_subword.txt"
        out_path.write_text("\n".join(encoded), encoding="utf-8")
        print(f"[done] Encoded {split} → {out_path}")

    print("[all done] Subword model + splits ready.")

if __name__ == "__main__":
    run()
