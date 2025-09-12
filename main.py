# main.py
import torch
from train import train_model
import random
import numpy as np

from models import LSTMLM, TransformerLM
import data_prep
import build_vocab_subword
import build_vocab_word
from generate_text import generate_text


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
    #data_prep.run()
    #build_vocab_word.run()
    #build_vocab_subword.run()

    # --------------------
    # 2. Train models
    # --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_lstm = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.3, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "baseline_lstm",
        "model_class": "LSTMLM"  
    }

    baseline_trf = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "num_heads": 4,
        "num_layers": 2, "ff_hidden": 64,
        "dropout": 0.3, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "baseline_trf",
        "model_class": "TransformerLM"   
    }

    print("=== Training baseline LSTM ===")
    #train_model(baseline_lstm)

    #generate_text(baseline_lstm, temperature=0.7, start_word="marriage")
    #generate_text(baseline_lstm, temperature=1.0, start_word="marriage")
    #generate_text(baseline_lstm, temperature=1.3, start_word="marriage")

    print("=== Training baseline Transformer ===")
    train_model(baseline_trf)

    generate_text(baseline_trf, temperature=0.7, start_word="marriage")
    generate_text(baseline_trf, temperature=1.0, start_word="marriage")
    generate_text(baseline_trf, temperature=1.3, start_word="marriage")

    # --------------------
    # 3. Ablation studies - LSTMLM
    # --------------------

    # dropout = 0.0 vs dropout = 0.2
    dropout00_lstm = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.0, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "dropout00_lstm",
        "model_class": "LSTMLM"            
    }

    dropout02_lstm = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.2, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "dropout02_lstm",
        "model_class": "LSTMLM"   
    }

    # tokenization: word (baseline) vs subword
    subword_lstm = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "hidden_size": 64, "num_layers": 1,
        "dropout": 0.3, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "subword.json",
        "name": "subword_lstm",
        "model_class": "LSTMLM"   
    }

    print("=== Ablation studies: LSTM ===")

    print("Training LSTMLM on dropout = 0.0")
    #train_model(dropout00_lstm)
    #generate_text(dropout00_lstm, temperature=0.7, start_word="marriage")
    #generate_text(dropout00_lstm, temperature=1.0, start_word="marriage")
    #generate_text(dropout00_lstm, temperature=1.3, start_word="marriage")

    print("Training LSTMLM on dropout = 0.2")
    #train_model(dropout02_lstm)
    #generate_text(dropout02_lstm, temperature=0.7, start_word="marriage")
    #generate_text(dropout02_lstm, temperature=1.0, start_word="marriage")
    #generate_text(dropout02_lstm, temperature=1.3, start_word="marriage")

    print("Training LSTMLM on subword vocab")
    #train_model(subword_lstm)
    #generate_text(subword_lstm, temperature=0.7, start_word="marriage")
    #generate_text(subword_lstm, temperature=1.0, start_word="marriage")
    #generate_text(subword_lstm, temperature=1.3, start_word="marriage")

    # --------------------
    # 4. Ablation studies - Transformer
    # --------------------

    # dropout = 0.0 vs dropout = 0.2
    dropout00_trf = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "num_heads": 4,
        "num_layers": 2, "ff_hidden": 64,
        "dropout": 0.0, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "dropout00_trf",
        "model_class": "TransformerLM"   
    }

    dropout02_trf = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "num_heads": 4,
        "num_layers": 2, "ff_hidden": 64,
        "dropout": 0.2, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "vocab.json",
        "name": "dropout02_trf",
        "model_class": "TransformerLM"   
    }

    # tokenization: word (baseline) vs subword
    subword_trf = {
        "seq_len": 32, "batch_size": 64,
        "embed_size": 32, "num_heads": 4,
        "num_layers": 2, "ff_hidden": 64,
        "dropout": 0.0, "epochs": 10,
        "lr": 5e-4, "weight_decay": 1e-5,
        "device": device,
        "vocab_file": "subword.json",
        "name": "subword_trf",
        "model_class": "TransformerLM"   
    }
    print("=== Ablation studies: TransformerLM ===")

    print("Training TransformerLM on dropout = 0.0")
    #train_model(dropout00_trf)
    #generate_text(dropout00_trf, temperature=0.7, start_word="marriage")
    #generate_text(dropout00_trf, temperature=1.0, start_word="marriage")
    #generate_text(dropout00_trf, temperature=1.3, start_word="marriage")

    print("Training TransformerLM on dropout = 0.2")
    #train_model(dropout02_trf)
    #generate_text(dropout02_trf, temperature=0.7, start_word="marriage")
    #generate_text(dropout02_trf, temperature=1.0, start_word="marriage")
    #generate_text(dropout02_trf, temperature=1.3, start_word="marriage")

    print("Training TransformerLM on subword vocab")
    train_model(subword_trf)
    generate_text(subword_trf, temperature=0.7, start_word="marriage")
    generate_text(subword_trf, temperature=1.0, start_word="marriage")
    generate_text(subword_trf, temperature=1.3, start_word="marriage")


if __name__ == "__main__":
    main()
