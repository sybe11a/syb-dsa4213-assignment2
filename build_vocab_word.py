"""
Build Word Vocab (from TRAIN split only)
----------------------------------------
- Reads split/train.txt (sentences with <s> ... </s>)
- Builds word2id and frequency dict
- Saves:
    data/vocab.json
    data/freqs_train.json
"""

from __future__ import annotations
from pathlib import Path
import json, collections

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "split" / "train.txt"
VOCAB_JSON = DATA_DIR / "vocab.json"
FREQS_JSON = DATA_DIR / "freqs_train.json"
MIN_COUNT = 1

def read_sentences(path: Path):
    return [line.strip().split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def build_vocab(train_sentences, min_count=1):
    freqs = collections.Counter()
    for toks in train_sentences: freqs.update(toks)
    vocab_items = [(w,c) for w,c in freqs.items() if c>=min_count]
    vocab_items.sort(key=lambda x: (-x[1], x[0]))
    word2id = {w:i for i,(w,_) in enumerate(vocab_items)}
    return word2id, dict(freqs)

def run():
    train_sents = read_sentences(TRAIN_PATH)
    word2id, freqs = build_vocab(train_sents, MIN_COUNT)
    VOCAB_JSON.write_text(json.dumps(word2id, indent=2, ensure_ascii=False), encoding="utf-8")
    FREQS_JSON.write_text(json.dumps(freqs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Vocab built. Size={len(word2id)}. Saved to vocab.json, freqs_train.json.")

if __name__ == "__main__":
    run()
