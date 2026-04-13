"""
Data preprocessing module.

Implements:
  1. Text cleaning (remove HTML, special characters, normalize whitespace)
  2. Dataset segmentation: train (70%) / test (20%) / val (10%)
  3. Corpus statistics

"""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR, TRAIN_RATIO, TEST_RATIO, VAL_RATIO


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_NORMALIZE_RE = re.compile(r"[　]+")   # Full-width spaces


def clean_text(text: str) -> str:
    """
    Clean raw text:
      - Strip HTML tags
      - Remove URLs
      - Normalize whitespace (including full-width spaces)
      - Strip leading/trailing whitespace
    """
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PUNCT_NORMALIZE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """Convert Chinese punctuation to uniform forms."""
    mapping = {
        "，": "，",
        "。": "。",
        "！": "！",
        "？": "？",
        "；": "；",
        "：": "：",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for src, tgt in mapping.items():
        text = text.replace(src, tgt)
    return text


def is_valid_sample(text: str, min_chars: int = 10) -> bool:
    """Filter out samples that are too short or empty."""
    return len(text.strip()) >= min_chars


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    samples: List[Dict[str, Any]],
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split samples into train / test / val sets using the ratios from the paper.

    Args:
        samples: List of data samples (dicts).
        train_ratio: Fraction for training (default 0.70).
        test_ratio:  Fraction for testing  (default 0.20).
        val_ratio:   Fraction for validation (default 0.10).
        seed:        Random seed for reproducibility.

    Returns:
        (train_set, test_set, val_set)
    """
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    data = list(samples)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)

    train = data[:n_train]
    test = data[n_train: n_train + n_test]
    val = data[n_train + n_test:]
    return train, test, val


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_corpus(
    raw_records: List[Dict[str, Any]],
    text_field: str = "content",
) -> List[Dict[str, Any]]:
    """
    Apply full preprocessing to a list of raw records:
      1. Clean text
      2. Normalize punctuation
      3. Filter invalid samples

    Args:
        raw_records: Records as returned by data_collection.load_raw_data().
        text_field:  Key in each record that holds the main text.

    Returns:
        Cleaned and filtered records.
    """
    cleaned = []
    for record in raw_records:
        text = record.get(text_field, "")
        text = clean_text(text)
        text = normalize_punctuation(text)
        if not is_valid_sample(text):
            continue
        record = dict(record)
        record[text_field] = text
        # Also clean title
        if "title" in record:
            record["title"] = clean_text(normalize_punctuation(record["title"]))
        cleaned.append(record)
    return cleaned


# ---------------------------------------------------------------------------
# NER data utilities
# ---------------------------------------------------------------------------

def load_ner_data(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Load NER dataset in CoNLL-2003 format (space-separated token + label).

    Format:
        字 B-PRODUCT
        体 I-PRODUCT
        沙 B-MATERIAL
        ...

    Returns:
        List of (tokens, labels) tuples (one per sentence).
    """
    sentences = []
    tokens, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])
    if tokens:
        sentences.append((tokens, labels))
    return sentences


def save_ner_data(
    sentences: List[Tuple[List[str], List[str]]],
    file_path: str,
) -> None:
    """Save NER data in CoNLL format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for tok, lbl in zip(tokens, labels):
                f.write(f"{tok} {lbl}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# STS-B style similarity data utilities
# ---------------------------------------------------------------------------

def load_similarity_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load sentence-pair similarity data in STS-B format.

    Expected TSV columns: index, genre, filename, year, old_index,
    source1, source2, sentence1, sentence2, score

    Returns:
        List of dicts with keys: sentence1, sentence2, score (float 0–5).
    """
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("index"):
                continue   # Skip header
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            sentence1 = parts[-3]
            sentence2 = parts[-2]
            try:
                score = float(parts[-1])
            except ValueError:
                continue
            pairs.append({"sentence1": sentence1, "sentence2": sentence2, "score": score})
    return pairs


def save_similarity_data(
    pairs: List[Dict[str, Any]],
    file_path: str,
) -> None:
    """Save similarity pairs as TSV (STS-B format)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("index\tsentence1\tsentence2\tscore\n")
        for i, pair in enumerate(pairs):
            f.write(
                f"{i}\t{pair['sentence1']}\t{pair['sentence2']}\t{pair['score']:.4f}\n"
            )


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------

def corpus_statistics(samples: List[Dict[str, Any]], text_field: str = "content") -> Dict[str, Any]:
    """Compute basic statistics over a text corpus."""
    lengths = [len(s.get(text_field, "")) for s in samples]
    categories = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    return {
        "total_samples": len(samples),
        "avg_char_length": sum(lengths) / max(len(lengths), 1),
        "min_char_length": min(lengths) if lengths else 0,
        "max_char_length": max(lengths) if lengths else 0,
        "category_distribution": categories,
    }


# ---------------------------------------------------------------------------
# Main preprocessing runner
# ---------------------------------------------------------------------------

def run_preprocessing(
    raw_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Full preprocessing pipeline:
      1. Load raw data
      2. Clean and filter
      3. Split into train/val/test
      4. Save splits to disk

    Args:
        raw_data_path: Path to raw JSONL file from crawler.
        output_dir:    Directory to write split files.

    Returns:
        Dict mapping split name to output file path.
    """
    raw_data_path = raw_data_path or os.path.join(DATA_DIR, "raw_data.jsonl")
    output_dir = output_dir or os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Load
    records = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[Preprocessing] Loaded {len(records)} raw records.")

    # Clean
    records = preprocess_corpus(records)
    print(f"[Preprocessing] After cleaning: {len(records)} records.")

    # Statistics
    stats = corpus_statistics(records)
    print(f"[Preprocessing] Statistics: {stats}")

    # Split
    train, test, val = split_dataset(records)
    print(
        f"[Preprocessing] Split → train={len(train)}, "
        f"test={len(test)}, val={len(val)}"
    )

    # Save
    paths = {}
    for split_name, split_data in [("train", train), ("test", test), ("val", val)]:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths[split_name] = path
        print(f"[Preprocessing] Saved {split_name} → {path}")

    # Save statistics
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return paths
