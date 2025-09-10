"""
NLM Data Prep Module (Pride & Prejudice - pg1342)
-------------------------------------------------
- Cleans raw Gutenberg text
- Normalizes aliases
- Tokenizes by sentence with stopword filtering (keeps mr/mrs/miss; drops reporting verbs)
- Adds <s> ... </s> markers around every sentence
- Creates reproducible train/val/test split (by sentence)
- Saves:
    data/
      tokens_sentences.txt              (all sentences, each line with <s> ... </s>)
      tokens_flat.txt                   (flat stream, markers included)
      split/                            (split-specific artifacts)
        train.txt
        val.txt
        test.txt
        train_flat.txt
        val_flat.txt
        test_flat.txt
        split_meta.json
      vocab.json                        (built from TRAIN only; min_count=1)
      freqs_train.json                  (TRAIN word counts >=1)
"""

from __future__ import annotations
from pathlib import Path
import re, json, collections, random
from typing import List, Dict
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "pg1342_raw.txt"

TOKENS_SENT_PATH = DATA_DIR / "tokens_sentences.txt"
TOKENS_FLAT_PATH = DATA_DIR / "tokens_flat.txt"

SPLIT_DIR = DATA_DIR / "split"
SPLIT_SENT = {
    "train": SPLIT_DIR / "train.txt",
    "val":   SPLIT_DIR / "val.txt",
    "test":  SPLIT_DIR / "test.txt",
}
SPLIT_FLAT = {
    "train": SPLIT_DIR / "train_flat.txt",
    "val":   SPLIT_DIR / "val_flat.txt",
    "test":  SPLIT_DIR / "test_flat.txt",
}
SPLIT_META = SPLIT_DIR / "split_meta.json"

VOCAB_JSON = DATA_DIR / "vocab.json"
FREQS_TRAIN_JSON = DATA_DIR / "freqs_train.json"

RATIOS = (0.80, 0.10, 0.10)
SEED = 42
MIN_COUNT = 1

# -----------------------------
# Helpers
# -----------------------------
def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def _read_raw() -> str:
    raw = RAW_PATH.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return unidecode(raw)

def _crop_gutenberg(raw: str) -> str:
    start_pat = re.compile(
        r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        re.IGNORECASE | re.DOTALL
    )
    end_pat = re.compile(
        r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        re.IGNORECASE | re.DOTALL
    )
    sm, em = start_pat.search(raw), end_pat.search(raw)
    return raw[sm.end():em.start()] if (sm and em) else raw

def _clean_core(core: str) -> str:
    core = re.sub(r"\[\s*Illustration:?[\s\S]*?\]", "", core, flags=re.IGNORECASE)
    core = re.sub(r'(?m)^\s*[\[\]]+\s*$', '', core)
    core = core.replace('[', '').replace(']', '')
    core = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", "", core)
    core = re.sub(r"\bNIND\b", "", core)

    incipit_pat = re.compile(r"\bit is a truth universally acknowledged\b", re.IGNORECASE)
    incipit_m = incipit_pat.search(core)
    if incipit_m:
        core = core[incipit_m.start():]

    core = re.sub(r"\n{3,}", "\n\n", core).strip()
    core = re.sub(r"\s*CHISWICK PRESS[\s\S]*$", "", core, flags=re.IGNORECASE)
    return core

def _apply_aliases(text: str) -> str:
    ALIASES: Dict[str, str] = {
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
    normalized = text
    for pat, repl in ALIASES.items():
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)
    return normalized

def _tokenize_sentences(text: str) -> List[List[str]]:
    text = re.sub(r"\bMr\.", "Mr", text)
    text = re.sub(r"\bMrs\.", "Mrs", text)
    HEADING_RE = re.compile(r"(?mi)^\s*chapter\s+[ivxlcdm]+\b[^\n]*\n?")
    work_text = HEADING_RE.sub("", text)

    sentences_raw = [s.strip() for s in re.split(r'(?<=[.!?])\s+', work_text) if s.strip()]

    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    STOP_KEEP = {"mr", "mrs", "miss"}
    REPORTING_VERBS = {
        "said","replied","asked","cried","exclaimed","answered",
        "returned","observed","continued","remarked","added",
        "interrupted","retorted","rejoined","repeated","inquired",
        "whispered","murmured"
    }
    stop = {w for w in set(stopwords.words("english")) if w not in STOP_KEEP}
    stop |= REPORTING_VERBS

    TOKEN_RE = re.compile(r"[a-z](?:[a-z_]*[a-z])?")

    def tok(s: str):
        s = s.lower()
        tokens = TOKEN_RE.findall(s)
        return [t for t in tokens if t and t not in stop]

    sentences_tokens = [tok(s) for s in sentences_raw]
    sentences_tokens = [toks for toks in sentences_tokens if toks]
    # add sentence markers
    sentences_tokens = [["<s>"] + toks + ["</s>"] for toks in sentences_tokens]
    return sentences_tokens

def _save_sentences_and_flat(sentences_tokens: List[List[str]], path_sent, path_flat) -> None:
    with open(path_sent, "w", encoding="utf-8") as f:
        for toks in sentences_tokens:
            f.write(" ".join(toks) + "\n")
    with open(path_flat, "w", encoding="utf-8") as f:
        for toks in sentences_tokens:
            f.write(" ".join(toks) + " ")

def _split_sentences(sentences_tokens: List[List[str]], ratios=(0.8,0.1,0.1), seed=42):
    n = len(sentences_tokens)
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    n_test  = n - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train+n_val]
    test_idx  = idxs[n_train+n_val:]

    def take(indices): return [sentences_tokens[i] for i in indices]
    return take(train_idx), take(val_idx), take(test_idx), {
        "total_sentences": n,
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "seed": seed,
    }

def _build_vocab_from_train(train_sentences: List[List[str]], min_count=1):
    freqs = collections.Counter()
    for toks in train_sentences:
        freqs.update(toks)
    vocab_items = [(w, c) for w, c in freqs.items() if c >= min_count]
    vocab_items.sort(key=lambda x: (-x[1], x[0]))
    word2id = {w: i for i, (w, _) in enumerate(vocab_items)}
    return word2id, dict(freqs)

# -----------------------------
# Main
# -----------------------------
def run() -> None:
    _ensure_dirs()
    raw = _read_raw()
    core = _crop_gutenberg(raw)
    core = _clean_core(core)
    normalized = _apply_aliases(core)
    sentences_tokens = _tokenize_sentences(normalized)

    # save all
    _save_sentences_and_flat(sentences_tokens, TOKENS_SENT_PATH, TOKENS_FLAT_PATH)

    # split
    train_s, val_s, test_s, meta = _split_sentences(sentences_tokens, RATIOS, SEED)
    _save_sentences_and_flat(train_s, SPLIT_SENT["train"], SPLIT_FLAT["train"])
    _save_sentences_and_flat(val_s,   SPLIT_SENT["val"],   SPLIT_FLAT["val"])
    _save_sentences_and_flat(test_s,  SPLIT_SENT["test"],  SPLIT_FLAT["test"])
    SPLIT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # vocab (train only)
    word2id, freqs_train = _build_vocab_from_train(train_s, MIN_COUNT)
    VOCAB_JSON.write_text(json.dumps(word2id, ensure_ascii=False, indent=2), encoding="utf-8")
    FREQS_TRAIN_JSON.write_text(json.dumps(freqs_train, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Preprocessing done. Outputs saved under data/ and data/split/")

if __name__ == "__main__":
    run()
