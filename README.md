# syb-dsa4213-assignment2


# DSA4213 Assignment 2 – Small Language Models on *Pride and Prejudice*

This repository contains the code, data preprocessing, and experiments for Assignment 2 of DSA4213. The project implements and compares small sequence models — an LSTM and a lightweight Transformer — on a language modeling task using *Pride and Prejudice* (Project Gutenberg, ID 1342).

## Repository Structure

```
.
├── artifacts/                 # Saved models, loss plots, and generated text
├── data/                      
│   ├── pg1342_raw.txt         # Raw Project Gutenberg text
│   ├── tokens_sentences.txt   # Sentence-tokenized text with <s> and </s>
│   ├── tokens_flat.txt        # Flattened tokens
│   ├── vocab.json             # Word-level vocabulary (token → id)
│   ├── subword.*              # Subword model, vocab, and mapping
│   ├── split/                 # Train/val/test splits
│   │   ├── *_flat.txt         # Word-level splits
│   │   ├── *_subword.txt      # Subword-encoded splits
│   │   └── split_meta.json    # Metadata (sizes of splits)
│   └── freqs_train.json       # Training frequency counts
│
├── build_vocab_word.py        # Build word-level vocabulary
├── build_vocab_subword.py     # Train SentencePiece and build subword vocab
├── data_prep.py               # Clean and preprocess raw text into splits
├── models.py                  # Model definitions: LSTM and Transformer
├── train.py                   # Training loop with logging, checkpointing, plots
├── generate_text.py           # Unified text generation (LSTM & Transformer)
├── main.py                    # Entry point: run preprocessing, training, ablations
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup

1. Clone this repo and set up a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   pip install -r requirements.txt
   ```

2. Ensure you have **PyTorch** and **SentencePiece** installed.

3. Download the raw dataset (*Pride and Prejudice*) from Project Gutenberg (already included here as `pg1342_raw.txt`).

---

## Running the Code

### 1. Data Preprocessing

Run preprocessing to clean the raw text, normalize aliases, and create splits:

```bash
python data_prep.py
python build_vocab_word.py
python build_vocab_subword.py
```

### 2. Training Baselines

Train baseline models:

```bash
python main.py
```

This will:

* Train LSTM and Transformer baselines
* Run ablation studies (dropout, tokenization)
* Save checkpoints in `artifacts/`

### 3. Text Generation

Generate text from trained models:

```bash
python generate_text.py
```

Outputs are saved in `artifacts/*.txt`, with sampling at T = 0.7, 1.0, 1.3.

---

## Results

* **Baselines:**

  * LSTM: Overfit quickly, test perplexity \~927
  * Transformer: Generalized strongly, test perplexity \~4.3
* **Ablations:**

  * Dropout: Minor effects on generalization
  * Subword tokenization: Reduced sparsity, but outputs showed unnatural fragments

---

## Notes

* Models are intentionally small (hidden sizes 32–64, 1–2 layers) to highlight trade-offs between generalization and text generation quality on limited data.
* Perplexity is reported on validation and test sets, but qualitative analysis (sampled text) is essential to evaluate generative performance.
