# Data

The raw corpus and annotated datasets used in the paper are **not included** in this
repository due to a confidentiality agreement with the collaborating industrial partner
(a high-end integrated home furnishing enterprise).

## What is missing

| File / Directory | Description |
|---|---|
| `data_files/raw/raw_corpus.jsonl` | Web-crawled product descriptions and design articles |
| `data_files/processed/train.conll` | CoNLL-format NER training set (70% split) |
| `data_files/processed/val.conll` | CoNLL-format NER validation set (10% split) |
| `data_files/processed/test.conll` | CoNLL-format NER test set (20% split) |
| `data_files/processed/qa_train.json` | Q&A training samples (text + NER labels + intent) |
| `data_files/processed/val_sts.tsv` | STS-B format similarity validation pairs |
| `data_files/processed/test_sts.tsv` | STS-B format similarity test pairs |
| `data_files/lexicon.txt` | Domain lexicon for LEBERT (one word per line) |

## Expected formats

**NER data** (`train.conll`, `val.conll`, `test.conll`) — CoNLL-2003 format:
```
北  B-STYLE
欧  I-STYLE
风  I-STYLE
格  I-STYLE
沙  B-PRODUCT
发  I-PRODUCT
的  O
```
Empty line separates sentences.

**Q&A data** (`qa_train.json`) — JSON list:
```json
[
  {
    "text": "客厅用什么风格的沙发",
    "ner_labels": ["B-SPACE", "I-SPACE", "O", "O", "O", "O", "O", "B-PRODUCT", "I-PRODUCT"],
    "intent": "design_style"
  }
]
```

**Similarity pairs** (`val_sts.tsv`, `test_sts.tsv`) — tab-separated:
```
sentence1\tsentence2\tscore
北欧风格客厅设计\t现代简约卧室装修\t2.5
```
Score range: 0.0 (unrelated) – 5.0 (equivalent).

**Domain lexicon** (`lexicon.txt`) — one term per line:
```
实木
大理石
北欧风格
```

## Contact

For data access requests, contact the corresponding author:
**Sa Guohua** — Zhejiang University, Department of Mechanical Engineering
