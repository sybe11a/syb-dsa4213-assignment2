# DSA4213 Assignment 2 – Small Language Models on *Pride and Prejudice*

This repository contains the code, data preprocessing, and experiments for Assignment 2 of DSA4213. The project implements and compares small sequence models — an LSTM and a lightweight Transformer — on a language modeling task using *Pride and Prejudice* (Project Gutenberg, ID 1342).

In addition to baseline training, the project includes **ablation studies** to analyze model behavior:

* **Dropout:** comparing 0.0 vs 0.2
* **Tokenization:** comparing word-level vocabulary vs subword vocabulary (SentencePiece BPE)

---

## Repository Structure

```
.
├── artifacts/                 # Saved models, loss plots, generated text
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

---
## Setup

1. Clone the repo and set up a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

2. Ensure you have **PyTorch** and **SentencePiece** installed (see `requirements.txt`).

3. Dataset (*Pride and Prejudice*) is already included as `data/pg1342_raw.txt`.

---

## Core Modules

* **`data_prep.py`**
  Cleans the raw text, applies alias normalization, sentence tokenization, and creates train/val/test splits.

* **`build_vocab_word.py` / `build_vocab_subword.py`**
  Construct vocabularies. Word-level vocab is a simple token→id map; subword vocab is trained with SentencePiece BPE.

* **`models.py`**
  Defines the model architectures:

  * `LSTMLM`: single-layer LSTM language model
  * `TransformerLM`: small Transformer encoder-decoder variant for next-token prediction

* **`train.py`**
  Contains the training loop (cross-entropy loss, perplexity, validation checks, checkpoint saving, loss plotting).

* **`generate_text.py`**
  Unified sampling interface for both models. Supports temperature-based sampling at different values (0.7, 1.0, 1.3).

* **`main.py`**
  Orchestrates the full workflow: preprocessing, vocabulary building, baseline training, ablation studies, and text generation.

---

## Running the Code

Run the full pipeline:

```bash
python main.py
```

This will:

* Preprocess the raw text into cleaned token sequences and splits
* Build both word and subword vocabularies
* Train LSTM and Transformer baselines
* Run ablation studies (dropout, tokenization)
* Save checkpoints, training curves, and generated text into `artifacts/`

---

## Output Files

For each experiment (e.g., `baseline_lstm`, `baseline_trf`, `dropout00_trf`, `subword_trf`), the system generates:

* **Training curves:**
  `{experiment}_loss.png`

* **Text samples:**
  `{experiment}_T{temperature}.txt`
  (e.g., `baseline_trf_T1.0.txt`)

* **Model checkpoints:**
  `{experiment}_best.pt`

---
## Notes

* Models are intentionally small (hidden sizes 32–64, 1–2 layers) to highlight trade-offs between generalization and generation quality on limited data.
* Perplexity is reported on validation/test sets, but qualitative text generation is essential to assess performance.
