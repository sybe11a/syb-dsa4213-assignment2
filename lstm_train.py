# Minimal word-level LSTM LM for your pg1342 data
# Adds: val/test perplexity, a simple loss plot, text sampling (T=0.7,1.0,1.3), training time.

import json, math, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------
# Paths (from your prep step)
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
PLOT_PATH = ARTIFACTS_DIR / "lstm_loss.png"

# --------------------
# Hyperparams (simple, within guide)
# --------------------
SEQ_LEN       = 32
BATCH_SIZE    = 64
EMBED_SIZE    = 64
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
LEARNING_RATE = 5e-4
EPOCHS        = 10
DROPOUT       = 0.3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DECAY  =1e-5

# --------------------
# IO helpers
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

# --------------------
# Dataset (fixed sliding window, like your reference)
# --------------------
class FlatStreamDataset(Dataset):
    def __init__(self, ids, seq_len):
        assert len(ids) > seq_len, "Not enough tokens for chosen seq_len."
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
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)   
        self.fc   = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)              
        logits = self.fc(out)
        return logits, hidden

# --------------------
# Eval + Sampling
# --------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb)
        loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
        tok = yb.numel()
        tot_loss += loss.item() * tok
        tot_tokens += tok
    avg_loss = tot_loss / max(1, tot_tokens)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# --------------------
# Main
# --------------------
def main():
    # Load vocab & data
    w2i = load_vocab(VOCAB_JSON)
    i2w = {i: w for w, i in w2i.items()}
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

    model = LSTMLM(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        #  Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "artifacts/lstm_best.pt")
            print(f"  ↳ Saved new best model (epoch {epoch}, val_loss={val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"Training time: {total_time:.1f}s")

    # Test (final)
    test_loss, test_ppl = evaluate(model, test_loader, criterion, DEVICE)
    print(f"[Test] loss={test_loss:.4f} ppl={test_ppl:.2f}")

    # loss plot (PNG)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, EPOCHS+1))
        plt.figure()
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, val_losses, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("LSTM LM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_PATH)
        print(f"Saved plot: {PLOT_PATH}")
    except Exception as e:
        print(f"(Plotting skipped: {e})")


if __name__ == "__main__":
    main()
