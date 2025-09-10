# transformer_train.py
# Minimal small Transformer LM (word-level) for DSA4213 Assignment 2

import json, math, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------
# Paths
# --------------------
DATA_DIR = Path("data")
SPLIT = {
    "train": DATA_DIR / "split" / "train_flat.txt",
    "val":   DATA_DIR / "split" / "val_flat.txt",
    "test":  DATA_DIR / "split" / "test_flat.txt",
}
VOCAB_JSON = DATA_DIR / "vocab.json"

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = ARTIFACTS_DIR / "transformer_loss.png"

# --------------------
# Hyperparams (small Transformer)
# --------------------
SEQ_LEN       = 32
BATCH_SIZE    = 64
EMBED_SIZE    = 32   # smaller than LSTM, to keep lightweight
NUM_HEADS     = 4
NUM_LAYERS    = 2
FF_HIDDEN     = 64
DROPOUT       = 0.3
LEARNING_RATE = 5e-4
EPOCHS        = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY  =1e-5

# --------------------
# Data utils
# --------------------
def read_tokens(path: Path):
    return path.read_text(encoding="utf-8").strip().split()

def load_vocab(path: Path):
    w2i = json.loads(path.read_text(encoding="utf-8"))
    if "<unk>" not in w2i:
        w2i["<unk>"] = max(w2i.values(), default=-1) + 1
    return w2i

def to_ids(tokens, w2i):
    unk = w2i["<unk>"]
    return [w2i.get(t, unk) for t in tokens]

class FlatStreamDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# --------------------
# Model
# --------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(SEQ_LEN, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embed(x) + self.pos_embed(pos)
        out = self.transformer(x)
        logits = self.fc(out)
        return logits

# --------------------
# Eval
# --------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
        tok = yb.numel()
        tot_loss += loss.item() * tok
        tot_tokens += tok
    avg_loss = tot_loss / max(1, tot_tokens)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# --------------------
# Train
# --------------------
def main():
    # Load vocab & data
    w2i = load_vocab(VOCAB_JSON)
    vocab_size = len(w2i)
    print(f"Vocab size: {vocab_size}")

    train_ids = to_ids(read_tokens(SPLIT["train"]), w2i)
    val_ids   = to_ids(read_tokens(SPLIT["val"]),   w2i)
    test_ids  = to_ids(read_tokens(SPLIT["test"]),  w2i)

    train_ds = FlatStreamDataset(train_ids, SEQ_LEN)
    val_ds   = FlatStreamDataset(val_ids,   SEQ_LEN)
    test_ds  = FlatStreamDataset(test_ids,  SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = TransformerLM(vocab_size, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, FF_HIDDEN, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training
    train_losses, val_losses = [], []
    start_time = time.time()
    best_val_loss = float("inf")  # track best val loss

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss, running_tok = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optim.step()

            tok = yb.numel()
            running_loss += loss.item() * tok
            running_tok += tok
            pbar.set_postfix(train_loss=f"{(running_loss/max(1,running_tok)):.4f}")

        train_avg = running_loss / max(1, running_tok)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, DEVICE)
        train_losses.append(train_avg)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch}] train_loss={train_avg:.4f} ppl={math.exp(train_avg):.2f} "
              f"| val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "artifacts/transformer_best.pt")
            print(f"  ↳ Saved new best model (epoch {epoch}, val_loss={val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"Training time: {total_time:.1f}s")

    # Final test eval
    test_loss, test_ppl = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[Test] loss={test_loss:.4f} ppl={test_ppl:.2f}")

    # Plot loss
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, EPOCHS+1))
        plt.figure()
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, val_losses, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Transformer LM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        print(f"Saved plot: {PLOT_PATH}")
    except Exception as e:
        print(f"(Plotting skipped: {e})")

if __name__ == "__main__":
    main()
