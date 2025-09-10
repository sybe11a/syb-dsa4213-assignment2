"""
Build Subword Vocab (SentencePiece, from TRAIN split)
-----------------------------------------------------
- Trains a SentencePiece model (BPE or unigram) on the TRAIN split
- Saves:
    data/subword.model     (SentencePiece model file)
    data/subword.vocab     (human-readable vocab)
    data/subword.json      (token->id dict, for easy PyTorch use)
"""

from __future__ import annotations
from pathlib import Path
import json
import sentencepiece as spm

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "split" / "train.txt"

# outputs
MODEL_PREFIX = str(DATA_DIR / "subword")
MODEL_FILE   = DATA_DIR / "subword.model"
VOCAB_FILE   = DATA_DIR / "subword.vocab"
VOCAB_JSON   = DATA_DIR / "subword.json"

# config
VOCAB_SIZE = 8000        # adjust depending on dataset size
MODEL_TYPE = "bpe"       # "bpe" or "unigram"

def run():
    # SentencePiece expects one sentence per line (already the case)
    # But it wants a *single file path*, so just use TRAIN_PATH
    print(f"[info] Training SentencePiece on {TRAIN_PATH} ...")
    spm.SentencePieceTrainer.Train(
        input=str(TRAIN_PATH),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=1.0,  # English text = 1.0
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1,  # special tokens
    )

    print(f"[done] Model written to {MODEL_FILE}, vocab to {VOCAB_FILE}")

    # Build JSON mapping {subword: id}
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_FILE))

    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    VOCAB_JSON.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Subword vocab saved to {VOCAB_JSON}")

if __name__ == "__main__":
    run()
