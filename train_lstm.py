#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train script for LSTM LM (word-level) using your preprocessed outputs:
  - data/tokens_with_markers.txt
  - data/vocab.json

Saves:
  - checkpoints/lstm_lm_best.pt
  - checkpoints/config.json
Prints:
  - train/val perplexities per epoch
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path
import argparse
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lstm_lm import LSTMLanguageModel, detach_hidden, evaluate_perplexity, generate_text


# ---------------------------
# Data utils
# ---------------------------
def load_tokens_and_vocab(data_dir: Path) -> Tuple[List[str], Dict[str, int]]:
    tok_path = data_dir / "tokens_with_markers.txt"
    voc_path = data_dir / "vocab.json"
    tokens = tok_path.read_text(encoding="utf-8").split()
    word2id = json.loads(voc_path.read_text(encoding="utf-8"))
    return tokens, word2id


def tokens_to_ids(tokens: List[str], word2id: Dict[str, int]) -> List[int]:
    # OOVs should be none if you used min_count during preprocessing; otherwise you could map to <unk>
    ids = [word2id[t] for t in tokens if t in word2id]
    return ids


def split_stream(ids: List[int], train_ratio=0.8, valid_ratio=0.1, seed=42) -> Tuple[List[int], List[int], List[int]]:
    """
    For language modeling on a continuous stream, a simple contiguous split is fine.
    If you want to avoid splitting inside a sentence, you could scan for </s> boundaries.
    """
    N = len(ids)
    train_end = int(train_ratio * N)
    valid_end = int((train_ratio + valid_ratio) * N)
    return ids[:train_end], ids[train_end:valid_end], ids[valid_end:]


class TokenStreamDataset(Dataset):
    """Sliding window dataset over a flat id stream."""
    def __init__(self, id_list: List[int], seq_len: int):
        self.ids = torch.tensor(id_list, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ---------------------------
# Training / Eval
# ---------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    vocab_size: int,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        logits, _ = model(x, None)  # fresh hidden per batch; simple & stable
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        # gradient clipping helps LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def evaluate_ce(
    model: nn.Module,
    loader: DataLoader,
    vocab_size: int,
    device: torch.device
) -> float:
    """
    Return average cross-entropy over the loader (not perplexity).
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x, None)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1)


def save_checkpoint(
    out_dir: Path,
    model: nn.Module,
    config: dict,
    tag: str = "best"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"lstm_lm_{tag}.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------
# Orchestrator
# ---------------------------
def run_training(
    data_dir: Path,
    out_dir: Path,
    seq_len: int = 128,
    batch_size: int = 64,
    embed_size: int = 128,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 10,
    seed: int = 42,
    sample_every: int = 2,
    sample_prompt: str = "<s>",
    temperature: float = 1.0,
):
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    tokens, word2id = load_tokens_and_vocab(data_dir)
    ids = tokens_to_ids(tokens, word2id)
    train_ids, valid_ids, test_ids = split_stream(ids, train_ratio=0.8, valid_ratio=0.1, seed=seed)

    # Datasets / loaders
    train_ds = TokenStreamDataset(train_ids, seq_len)
    valid_ds = TokenStreamDataset(valid_ids, seq_len)
    test_ds  = TokenStreamDataset(test_ids,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # Model
    vocab_size = len(word2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reverse map for sampling
    id2word = {i: w for w, i in word2id.items()}
    start_tokens = [word2id[t] for t in sample_prompt.split() if t in word2id]

    best_val_ppl = float("inf")
    for ep in range(1, epochs + 1):
        train_ce = train_one_epoch(model, train_loader, optimizer, vocab_size, device)
        val_ce = evaluate_ce(model, valid_loader, vocab_size, device)
        train_ppl = math.exp(train_ce)
        val_ppl = math.exp(val_ce)

        print(f"[epoch {ep:02d}] train_ppl={train_ppl:.2f}  val_ppl={val_ppl:.2f}")

        # sample some text occasionally
        if sample_every and (ep % sample_every == 0) and start_tokens:
            s = generate_text(model, start_tokens, id2word, max_len=40, temperature=temperature, device=device)
            print("[sample]", s)

        # checkpoint best
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            cfg = dict(
                vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout, seq_len=seq_len, batch_size=batch_size,
                lr=lr, epochs=epochs, seed=seed
            )
            save_checkpoint(out_dir, model, cfg, tag="best")

    # Final test perplexity
    test_ppl = evaluate_perplexity(model, test_loader, vocab_size, device)
    print(f"[test] perplexity={test_ppl:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--embed_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_every", type=int, default=2)
    ap.add_argument("--sample_prompt", type=str, default="<s>")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    run_training(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
