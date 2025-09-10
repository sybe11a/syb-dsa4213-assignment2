# models.py
import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        return self.fc(out), hidden


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden, dropout, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_len, embed_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embed(x) + self.pos_embed(pos)
        out = self.transformer(x)
        return self.fc(out)
