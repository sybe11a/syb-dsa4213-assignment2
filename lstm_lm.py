#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM LM (word-level) for DSA4213 A2
- Defines the model, hidden-state helpers, and text generation utilities.
- Pure module (no I/O with dataset files).
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class LSTMLanguageModel(nn.Module):
    """
    A simple LSTM-based language model:
      Embedding -> LSTM (num_layers) -> Linear(vocab_size)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(hidden_size, vocab_size)

        # Initialize (optional but nice)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: LongTensor [B, T] of token ids
        hidden: (h0, c0) each [num_layers, B, hidden]
        returns:
          logits: [B, T, vocab_size]
          hidden: (hT, cT)
        """
        emb = self.embed(x)                    # [B, T, E]
        out, hidden = self.lstm(emb, hidden)   # [B, T, H]
        logits = self.proj(out)                # [B, T, V]
        return logits, hidden

    def init_hidden(self, batch_size: int, hidden_size: int, num_layers: int, device: torch.device):
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h0, c0)


def detach_hidden(hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detach hidden state from its history to avoid backprop through entire epoch."""
    h, c = hidden
    return (h.detach(), c.detach())


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    data_loader,
    vocab_size: int,
    device: torch.device
) -> float:
    """
    Compute cross-entropy over the loader and return perplexity = exp(CE).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")  # sum so we can average manually

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x, None)  # fresh hidden per batch
        # reshape to 2D
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
        total_tokens += y.numel()

    ce = total_loss / max(total_tokens, 1)
    ppl = math.exp(ce)
    return ppl


@torch.no_grad()
def generate_text(
    model: nn.Module,
    start_tokens: List[int],
    id2word: Dict[int, str],
    max_len: int = 50,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Temperature sampling from the model given a list of start token ids.
    Returns a space-joined string of tokens.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # [1, T0]
    x = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
    hidden = None
    generated: List[int] = list(start_tokens)

    for _ in range(max_len):
        logits, hidden = model(x, hidden)
        last_step = logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(last_step, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)

    # Map ids to surface tokens (space-joined; you can tweak formatting for <s>/</s>)
    toks = [id2word[i] for i in generated]
    return " ".join(toks)
