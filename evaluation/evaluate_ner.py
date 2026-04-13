"""
NER evaluation script.

Reproduces Table 2 (model comparison) and Table 10 (proposed model results):
  Proposed model: Acc=97.78%, P=93.85%, R=92.12%, F1=92.97%

Metrics (Section 3.2.2.1 / Equations 1–4):
  Acc = (TP + TN) / (P + N)
  P   = TP / (TP + FP)
  R   = TP / (TP + FN)
  F1  = 2PR / (P + R)

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 3.2.2.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import NER_CONFIG, NER_LABELS, MODEL_DIR, DATA_DIR
from data.data_preprocessing import load_ner_data
from knowledge_extraction.ner_model import (
    LEBERTBiLSTMAttentionCRF,
    NERDataset,
    NERTrainer,
    compare_ner_models,
)


# ---------------------------------------------------------------------------
# Token-level metrics (Equations 1–4)
# ---------------------------------------------------------------------------

def compute_metrics_token_level(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str],
    ignore_label: str = "O",
) -> Dict[str, float]:
    """
    Compute Accuracy, Precision, Recall, F1 at the token level.

    Matches Equations 1–4 in Section 3.2.2.1:
      TP: correctly predicted non-O labels
      FP: incorrectly predicted non-O labels
      FN: non-O labels missed by the model
      TN: correctly predicted O labels
    """
    tp = fp = fn = tn = 0
    total = correct = 0

    for pred_seq, lbl_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, lbl_seq):
            total += 1
            p_label = id2label.get(p, "O")
            l_label = id2label.get(l, "O")
            if p == l:
                correct += 1
            if l_label != ignore_label and p_label != ignore_label and p == l:
                tp += 1
            elif l_label != ignore_label and (p_label == ignore_label or p != l):
                fn += 1
            elif l_label == ignore_label and p_label != ignore_label:
                fp += 1
            else:
                tn += 1

    acc = correct / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Entity-level (span) metrics
# ---------------------------------------------------------------------------

def compute_metrics_entity_level(
    predictions: List[List[str]],
    labels: List[List[str]],
) -> Dict[str, float]:
    """
    Compute span-level entity F1 using seqeval (CoNLL-style evaluation).

    This is stricter than token-level: an entity is correct only if
    both its boundary and label match exactly.
    """
    try:
        from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        rec = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    except ImportError:
        print("[Eval] seqeval not installed. Falling back to token-level metrics.")
        label2id = {lbl: i for i, lbl in enumerate(NER_LABELS)}
        preds_ids = [[label2id.get(l, 0) for l in seq] for seq in predictions]
        labels_ids = [[label2id.get(l, 0) for l in seq] for seq in labels]
        id2label = {i: lbl for i, lbl in enumerate(NER_LABELS)}
        return compute_metrics_token_level(preds_ids, labels_ids, id2label)


# ---------------------------------------------------------------------------
# Per-entity-type breakdown
# ---------------------------------------------------------------------------

def compute_per_type_metrics(
    predictions: List[List[str]],
    labels: List[List[str]],
    entity_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute P/R/F1 for each entity type separately.

    Args:
        predictions: List of BIO label sequences (predicted).
        labels:      List of BIO label sequences (gold).
        entity_types: Entity types to evaluate (e.g., ["PRODUCT", "MATERIAL"]).

    Returns:
        {entity_type: {precision, recall, f1}}
    """
    if entity_types is None:
        entity_types = list({
            lbl.split("-")[-1]
            for seq in labels for lbl in seq
            if lbl != "O"
        })

    results = {}
    for etype in entity_types:
        tp = fp = fn = 0
        for pred_seq, lbl_seq in zip(predictions, labels):
            for p, l in zip(pred_seq, lbl_seq):
                p_type = p.split("-")[-1] if p != "O" else "O"
                l_type = l.split("-")[-1] if l != "O" else "O"
                if l_type == etype and p_type == etype and p == l:
                    tp += 1
                elif l_type == etype and (p_type != etype or p != l):
                    fn += 1
                elif l_type != etype and p_type == etype:
                    fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[etype] = {"precision": prec, "recall": rec, "f1": f1}
    return results


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_ner_model(
    model_dir: str,
    test_data_path: str,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load a trained NER model and evaluate on the test set.

    Args:
        model_dir:      Directory containing ner_model.pt.
        test_data_path: Path to CoNLL-format test file.
        output_path:    Optional JSON output path for results.
        device:         Torch device.

    Returns:
        Evaluation metrics dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(NER_CONFIG["bert_model"])

    # Load model
    model = LEBERTBiLSTMAttentionCRF(config=NER_CONFIG)
    trainer = NERTrainer(model, config=NER_CONFIG, device=device)
    trainer.load(model_dir)

    # Load test data
    test_sentences = load_ner_data(test_data_path)
    label2id = {lbl: i for i, lbl in enumerate(NER_LABELS)}
    test_dataset = NERDataset(test_sentences, tokenizer, label2id)
    test_loader = DataLoader(
        test_dataset, batch_size=NER_CONFIG["batch_size"], shuffle=False
    )

    # Evaluate
    metrics = trainer.evaluate(test_loader)
    print("\n[NER Evaluation Results]")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall:    {metrics['recall']*100:.2f}%")
    print(f"  F1:        {metrics['f1']*100:.2f}%")
    print()
    print("[Paper Target (Table 10)]: Acc=97.78, P=93.85, R=92.12, F1=92.97")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Model comparison (Table 2)
# ---------------------------------------------------------------------------

def print_model_comparison_table() -> None:
    """Print the NER model comparison from Table 2 of the paper."""
    table = compare_ner_models()
    print("\n[Table 2] NER Model Comparison")
    print(f"{'Model':<35} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 63)
    for model_name, scores in table.items():
        marker = " ★" if "LEBERT+BiLSTM" in model_name else ""
        print(
            f"{model_name + marker:<35} "
            f"{scores['precision']*100:>7.1f}% "
            f"{scores['recall']*100:>7.1f}% "
            f"{scores['f1']*100:>7.1f}%"
        )
    print("\n★ = Selected model (backbone for proposed joint extraction)")


if __name__ == "__main__":
    print_model_comparison_table()
