# train.py
import json, math, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------
# Dataset
# --------------------
class FlatStreamDataset(Dataset):
    def __init__(self, ids, seq_len):
        assert len(ids) > seq_len, "Not enough tokens for chosen seq_len."
        self.ids, self.seq_len = ids, seq_len
    def __len__(self): return len(self.ids) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y


# --------------------
# Helpers
# --------------------
def read_tokens(path: Path): 
    return path.read_text(encoding="utf-8").strip().split()

def load_vocab(path: Path):
    w2i = json.loads(path.read_text(encoding="utf-8"))
    if "<unk>" not in w2i: 
        w2i["<unk>"] = max(w2i.values(), default=-1) + 1
    return w2i

def to_ids(tokens, w2i): 
    return [w2i.get(t, w2i["<unk>"]) for t in tokens]

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); tot_loss, tot_tokens = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb) if isinstance(model(xb), torch.Tensor) else model(xb)[0]
        loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
        tot_loss += loss.item() * yb.numel(); tot_tokens += yb.numel()
    avg_loss = tot_loss / max(1, tot_tokens)
    return avg_loss, math.exp(avg_loss)


# --------------------
# Training entry point
# --------------------
def train_model(config: dict, model_class):
    from models import LSTMLM, TransformerLM  # lazy import

    DATA_DIR = Path("data")
    SPLIT = {k: DATA_DIR/"split"/f"{k}_flat.txt" for k in ["train","val","test"]}
    VOCAB_JSON = Path(DATA_DIR/config["vocab_file"])
    vocab_name = Path(config["vocab_file"]).stem  # e.g. "vocab" or "subword_vocab"
    ARTIFACTS_DIR = Path("artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS_DIR / f"{config['name']}_best.pt"
    plot_path = ARTIFACTS_DIR / f"{config['name']}_loss.png"

    # Load vocab & data
    w2i = load_vocab(VOCAB_JSON); vocab_size = len(w2i)
    train_ids = to_ids(read_tokens(SPLIT["train"]), w2i)
    val_ids   = to_ids(read_tokens(SPLIT["val"]),   w2i)
    test_ids  = to_ids(read_tokens(SPLIT["test"]),  w2i)

    train_loader = DataLoader(FlatStreamDataset(train_ids, config["seq_len"]),
                              batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(FlatStreamDataset(val_ids, config["seq_len"]),
                              batch_size=config["batch_size"])
    test_loader  = DataLoader(FlatStreamDataset(test_ids, config["seq_len"]),
                              batch_size=config["batch_size"])

    # Model init
    if model_class.__name__ == "LSTMLM":
        model = model_class(vocab_size, config["embed_size"], config["hidden_size"],
                            config["num_layers"], config["dropout"]).to(config["device"])
    elif model_class.__name__ == "TransformerLM":
        model = model_class(vocab_size, config["embed_size"], config["num_heads"],
                            config["num_layers"], config["ff_hidden"], 
                            config["dropout"], config["seq_len"]).to(config["device"])
    else:
        raise ValueError(f"Unknown model_class: {model_class}")

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Training loop
    train_losses, val_losses = [], []
    start_time = time.time(); best_val_loss = float("inf")

    for epoch in range(1, config["epochs"]+1):
        model.train()
        running_loss, running_tok = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for xb, yb in pbar:
            xb, yb = xb.to(config["device"]), yb.to(config["device"])
            optim.zero_grad()
            out = model(xb)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tok = yb.numel()
            running_loss += loss.item() * tok
            running_tok += tok
            pbar.set_postfix(train_loss=f"{(running_loss/max(1,running_tok)):.4f}")

        train_avg = running_loss / max(1, running_tok)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, config["device"])
        train_losses.append(train_avg); val_losses.append(val_loss)
        print(f"[Epoch {epoch}] train_loss={train_avg:.4f} ppl={math.exp(train_avg):.2f} "
              f"| val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ Saved new best model (epoch {epoch}, val_loss={val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"Training time: {total_time:.1f}s")

    # Final test on best model
    model.load_state_dict(torch.load(ckpt_path, map_location=config["device"]))
    test_loss, test_ppl = evaluate(model, test_loader, criterion, config["device"])
    print(f"[Test] loss={test_loss:.4f} ppl={test_ppl:.2f}")

    # Plot loss curve
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, config["epochs"]+1), train_losses, label="train")
        plt.plot(range(1, config["epochs"]+1), val_losses, label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{model_class.__name__}")
        plt.legend(); plt.tight_layout(); plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"(Plotting skipped: {e})")
