
"""
End-to-end data prep runner with modular functions.

Usage:
    python main.py --base . --incipit "it is a truth universally acknowledged"
"""

import argparse
from pathlib import Path

from nlm_data_prep.paths import default_data_paths
from nlm_data_prep.clean_gutenberg import clean_gutenberg_file
from nlm_data_prep.aliasing import normalize_aliases
from nlm_data_prep.aliasing_config import ALIASES
from nlm_data_prep.tokenize import (
    prepare_stopwords, split_sentences, tokenize_sentences, flatten_with_sentence_markers,
)
from nlm_data_prep.vocab import build_freqs, build_vocab, save_json
from nlm_data_prep.io_utils import write_tokens_as_space_separated

def run_pipeline(base: Path, incipit: str, stop_keep=None, reporting_verbs=None, min_count:int=3):
    paths = default_data_paths(base)
    RAW_PATH = paths["RAW_PATH"]
    CLEAN_PATH = paths["CLEAN_PATH"]
    NORM_PATH = paths["NORM_PATH"]
    TOKENS_WITH_MARKERS = paths["TOKENS_WITH_MARKERS"]
    VOCAB_JSON = paths["VOCAB_JSON"]
    FREQS_JSON = paths["FREQS_JSON"]

    # Step 1: Clean
    clean_text = clean_gutenberg_file(RAW_PATH, CLEAN_PATH, incipit_phrase=incipit)

    # Step 2: Aliasing
    normalized = normalize_aliases(clean_text, ALIASES)
    NORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORM_PATH.write_text(normalized, encoding="utf-8")

    # Step 3: Tokenize with sentence markers
    from nlm_data_prep.tokenize import DEFAULT_REPORTING_VERBS
    stop_keep = set(stop_keep or {"mr", "mrs", "miss"})
    reporting_verbs = set(reporting_verbs or DEFAULT_REPORTING_VERBS)
    stop = prepare_stopwords(stop_keep, reporting_verbs)

    sentences = split_sentences(normalized)
    sent_tokens = tokenize_sentences(sentences, stop)
    flat = flatten_with_sentence_markers(sent_tokens, start_tok="<s>", end_tok="</s>")
    write_tokens_as_space_separated(flat, TOKENS_WITH_MARKERS)

    # Step 4: Vocab
    freqs = build_freqs(sent_tokens)
    word2id, vocab_items = build_vocab(freqs, min_count=min_count)
    save_json(word2id, VOCAB_JSON)
    save_json(dict(vocab_items), FREQS_JSON)

    print(f"Done. Outputs written to {base.resolve() / 'data'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("."), help="Project root (will contain data/)")
    ap.add_argument("--incipit", type=str, default="it is a truth universally acknowledged", help="Phrase marking the story start")
    ap.add_argument("--min_count", type=int, default=3, help="Minimum frequency for vocab inclusion")
    args = ap.parse_args()
    run_pipeline(args.base, args.incipit, min_count=args.min_count)

if __name__ == "__main__":
    main()
