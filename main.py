# main.py
import torch
from train import train_model
import random
import numpy as np

from models import LSTMLM, TransformerLM
import data_prep
import build_vocab_subword
import build_vocab_word


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    # --------------------
    # 1. Run preprocessing
    # --------------------
    print("=== Running data preprocessing ===")
    data_prep.run()
    build_vocab_word.run()
    build_vocab_subword.run()

    # --------------------
    # 2. Train models
    # --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_lstm = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 64, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.3, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "baseline_lstm"  
    }

    baseline_trf = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "num_heads": 4,
        "num_layers": 2, "ff_hidden": 64,
        "dropout": 0.3, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "baseline_trf"
    }

    print("=== Training baseline LSTM ===")
    train_model(baseline_lstm, LSTMLM)

    print("=== Training baseline Transformer ===")
    train_model(baseline_trf, TransformerLM)

if __name__ == "__main__":
    main()
