"""
NLM Data Prep (Single File) - Pride & Prejudice (pg1342)

This script cleans the raw Gutenberg text, normalizes aliases, tokenizes with sentence
markers (<s>, </s>), and builds frequency & vocabulary JSONs.

Usage:
    python nlm_data_prep_single.py --base . \
        --incipit "it is a truth universally acknowledged" \
        --min_count 3

Outputs (under <base>/data):
    - pg1342_clean.txt
    - pg1342_normalized.txt
    - tokens_with_markers.txt      (flat stream of tokens with <s> ... </s>)
    - vocab.json
    - freqs.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

import sys 

# -----------------------------
# Paths helpers
# -----------------------------
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def default_data_paths(base: Path) -> Dict[str, Path]:
    base = base.resolve()
    data_dir = base / "data"
    return {
        "DATA_DIR": data_dir,
        "RAW_PATH": data_dir / "pg1342_raw.txt",
        "CLEAN_PATH": data_dir / "pg1342_clean.txt",
        "NORM_PATH": data_dir / "pg1342_normalized.txt",
        "TOKENS_WITH_MARKERS": data_dir / "tokens_with_markers.txt",
        "VOCAB_JSON": data_dir / "vocab.json",
        "FREQS_JSON": data_dir / "freqs.json",
    }

# -----------------------------
# Cleaning Gutenberg
# -----------------------------
GUT_START = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
GUT_END   = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",   re.I | re.S)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize_newlines_ascii(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return unidecode(raw)

def crop_gutenberg(raw: str) -> str:
    sm, em = GUT_START.search(raw), GUT_END.search(raw)
    return raw[sm.end():em.start()] if (sm and em) else raw

def strip_extraneous(core: str) -> str:
    # Remove multi-line [Illustration: ...] blocks
    core = re.sub(r"\[\s*Illustration:?[\s\S]*?\]", "", core, flags=re.I)
    # remove bracket-only lines
    core = re.sub(r'(?m)^\s*[\[\]]+\s*$', '', core)
    # strip stray brackets embedded in lines
    core = core.replace('[', '').replace(']', '')
    # Drop CSS-style comment blocks eg: /* NIND "..." */
    core = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", "", core)
    # Mop up any standalone NIND tokens that might remain
    core = re.sub(r"\bNIND\b", "", core)
    return core

def strip_preface_to_incipit(core: str, incipit_phrase: str) -> str:
    m = re.search(re.escape(incipit_phrase), core, flags=re.I)
    return core[m.start():] if m else core

def remove_trailing_printers_note(core: str) -> str:
    return re.sub(r"\s*CHISWICK PRESS[\s\S]*$", "", core, flags=re.I)

def tidy_whitespace(core: str) -> str:
    core = re.sub(r"\n{3,}", "\n\n", core).strip()
    return core

def clean_gutenberg_file(raw_path: Path, clean_path: Path, incipit_phrase: str) -> str:
    raw = read_text(raw_path)
    raw = normalize_newlines_ascii(raw)
    core = crop_gutenberg(raw)
    core = strip_extraneous(core)
    core = strip_preface_to_incipit(core, incipit_phrase)
    core = tidy_whitespace(core)
    core = remove_trailing_printers_note(core)
    ensure_parent(clean_path)
    clean_path.write_text(core, encoding="utf-8")
    return core

# -----------------------------
# Aliasing
# -----------------------------
# Canonical mapping (kept for reference/extension)
CANON = {
    "Elizabeth Bennet": "elizabeth_bennet",
    "Mr. Darcy": "mr_darcy",
    "Mrs. Bennet": "mrs_bennet",
    "Mr. Bennet": "mr_bennet",
    "Jane Bennet": "jane_bennet",
    "Mr. Bingley": "mr_bingley",
    "Miss Bingley": "miss_bingley",
    "Lady Catherine": "lady_catherine",
    "George Wickham": "george_wickham",
    "Charlotte Lucas": "charlotte_lucas",
    "Mr. Collins": "mr_collins",
    "Mr. Robinson": "mr_robinson",
    "Mrs. Long": "mrs_long",
    "Colonel Fitzwilliam": "colonel_fitzwilliam",
    "Georgiana Darcy": "georgiana_darcy",
}

ALIASES = {
    r"\bElizabeth\s+Bennet\b": "elizabeth_bennet",
    r"\bElizabeth\b": "elizabeth_bennet",
    r"\bEliza\s+Bennet\b": "elizabeth_bennet",
    r"\bLizzy\s+Bennet\b": "elizabeth_bennet",
    r"\bEliza\b": "elizabeth_bennet",
    r"\bLizzy\b": "elizabeth_bennet",
    r"\bMrs\.?\s+Darcy\b": "elizabeth_bennet",

    r"\bGeorgiana\s+Darcy\b": "georgiana_darcy",
    r"\bGeorgiana\b": "georgiana_darcy",
    r"\bMiss\s+Darcy\b": "georgiana_darcy",
    r"\bMr\.?\s+Darcy\b": "mr_darcy",
    r"\bFitzwilliam\s+Darcy\b": "mr_darcy",
    r"\bDarcy\b": "mr_darcy",

    r"\bMrs\.?\s+Bennet\b": "mrs_bennet",
    r"\bMr\.?\s+Bennet\b": "mr_bennet",
    r"\bJane\s+Bennet\b": "jane_bennet",
    r"\bJane\b": "jane_bennet",

    r"\bMr\.?\s+Bingley\b": "mr_bingley",
    r"\bMiss\s+Bingley\b": "miss_bingley",
    r"\bCaroline\s+Bingley\b": "miss_bingley",
    r"\bCaroline\b": "miss_bingley",

    r"\bColonel\s+Fitzwilliam\b": "colonel_fitzwilliam",
    r"\bLady\s+Catherine\s+de\s+Bourgh\b": "lady_catherine",
    r"\bLady\s+Catherine\b": "lady_catherine",
    r"\bGeorge\s+Wickham\b": "george_wickham",
    r"\bWickham\b": "george_wickham",
    r"\bMr\.?\s+Collins\b": "mr_collins",
    r"\bCharlotte\s+Lucas\b": "charlotte_lucas",
    r"\bMr\.?\s+Robinson\b": "mr_robinson",
    r"\bMrs\.?\s+Long\b": "mrs_long",
}

def normalize_aliases(text: str, aliases: Dict[str, str]) -> str:
    normalized = text
    # Apply longer/more specific patterns first
    for pat, repl in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        normalized = re.sub(pat, repl, normalized, flags=re.I)
    return normalized

# -----------------------------
# Tokenization
# -----------------------------
HEADING_RE = re.compile(r"(?mi)^\s*chapter\s+[ivxlcdm]+\b[^\n]*\n?")
TOKEN_RE = re.compile(r"[a-z](?:[a-z_]*[a-z])?")
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

DEFAULT_REPORTING_VERBS = {
    "said","replied","asked","cried","exclaimed","answered",
    "returned","observed","continued","remarked","added",
    "interrupted","retorted","rejoined","repeated","inquired",
    "whispered","murmured"
}

def prepare_stopwords(stop_keep: Set[str], reporting_verbs: Set[str]) -> Set[str]:
    nltk.download("stopwords", quiet=True)
    stop = set(stopwords.words("english"))
    stop = {w for w in stop if w not in stop_keep}
    stop |= reporting_verbs
    return stop

def normalize_honorifics(text: str) -> str:
    text = re.sub(r"\bMr\.", "Mr", text)
    text = re.sub(r"\bMrs\.", "Mrs", text)
    return text

def drop_headings(text: str) -> str:
    return HEADING_RE.sub("", text)

def split_sentences(text: str) -> List[str]:
    work_text = drop_headings(normalize_honorifics(text))
    return [s.strip() for s in SENT_SPLIT_RE.split(work_text) if s.strip()]

def tokenize_sentence(s: str, stop: Set[str]) -> List[str]:
    s = s.lower()
    tokens = TOKEN_RE.findall(s)  # preserves underscores (mr_darcy)
    return [t for t in tokens if t not in stop]

def tokenize_sentences(sentences: Iterable[str], stop: Set[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for s in sentences:
        toks = tokenize_sentence(s, stop)
        if toks:
            out.append(toks)
    return out

def flatten_with_sentence_markers(sent_tokens: List[List[str]], start_tok: str="<s>", end_tok: str="</s>") -> List[str]:
    flat: List[str] = []
    for toks in sent_tokens:
        flat.append(start_tok)
        flat.extend(toks)
        flat.append(end_tok)
    return flat

# -----------------------------
# Vocabulary
# -----------------------------
def build_freqs(sent_tokens: List[List[str]]) -> Counter:
    c = Counter()
    for toks in sent_tokens:
        c.update(toks)
    return c

def build_vocab(freqs: Counter, min_count: int = 3) -> Tuple[Dict[str,int], List[Tuple[str,int]]]:
    vocab_items = [(w, c) for w, c in freqs.items() if c >= min_count]
    vocab_items.sort(key=lambda x: (-x[1], x[0]))
    word2id = {w: i for i, (w, _) in enumerate(vocab_items)}
    return word2id, vocab_items

def save_json(obj, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Writers
# -----------------------------
def write_tokens_as_space_separated(tokens: List[str], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(" ".join(tokens))

# -----------------------------
# Orchestration
# -----------------------------
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
    ensure_parent(NORM_PATH)
    NORM_PATH.write_text(normalized, encoding="utf-8")

    # Step 3: Tokenize with sentence markers
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

    return {
        "clean_path": str(CLEAN_PATH),
        "normalized_path": str(NORM_PATH),
        "tokens_with_markers": str(TOKENS_WITH_MARKERS),
        "vocab_json": str(VOCAB_JSON),
        "freqs_json": str(FREQS_JSON),
        "num_sentences": len(sentences),
        "num_tokens_flat": len(flat),
        "vocab_size": len(word2id),
    }

# -----------------------------
# CLI
# -----------------------------
def main():
    base = Path("."); incipit = "it is a truth universally acknowledged"; min_count = 3
    print("[info] Running pipeline..."); stats = run_pipeline(base, incipit, min_count=min_count)
    print("[done] Outputs written to:", (base / "data").resolve()); print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    try: main()
    except Exception as e: print("[error]", e); sys.exit(1)

