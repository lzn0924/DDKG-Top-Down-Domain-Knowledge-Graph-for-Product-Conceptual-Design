"""
Joint Q&A model evaluation script.

Reproduces Table 10 results (proposed method):
  NER task:    Acc=97.78%, P=93.85%, R=92.12%, F1=92.97%
  Intent task: Acc=85.32%, WP=87.37%, WR=86.12%, WF1=86.73%

Evaluation metrics (Section 3.2.2.1):
  NER:    Accuracy (token), Precision, Recall, F1 (Equations 1–4)
  Intent: Accuracy, Weighted Precision (WP), Weighted Recall (WR),
          Weighted F1 (WF1) by category

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 3.2.2 and Table 10.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import NER_CONFIG, INTENT_LABELS, MODEL_DIR, DATA_DIR
from knowledge_application.qa_model import JointQAModel, QADataset, QATrainer


# ---------------------------------------------------------------------------
# Intent classification metrics (Table 10 – WP, WR, WF1)
# ---------------------------------------------------------------------------

def compute_intent_metrics(
    predictions: List[int],
    labels: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy and category-weighted P/R/F1 for intent classification.

    Matches Table 10 columns: Acc, WP, WR, WF1.
    """
    label_names = label_names or INTENT_LABELS
    acc = accuracy_score(labels, predictions)
    wp, wr, wf1, _ = precision_recall_fscore_support(
        labels, predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "intent_acc": acc,
        "intent_WP": wp,
        "intent_WR": wr,
        "intent_WF1": wf1,
    }


def compute_per_intent_metrics(
    predictions: List[int],
    labels: List[int],
    label_names: Optional[List[str]] = None,
) -> str:
    """Full per-class classification report for intent classification."""
    label_names = label_names or INTENT_LABELS
    return classification_report(
        labels, predictions,
        target_names=label_names,
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# NER metrics (Table 10 – Acc, P, R, F1)
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute token-level NER metrics (Equations 1–4).
    Matches Table 10 columns: Acc, P, R, F1.
    """
    tp = fp = fn = 0
    total = correct = 0
    for pred_seq, lbl_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, lbl_seq):
            total += 1
            p_lbl = id2label.get(p, "O")
            l_lbl = id2label.get(l, "O")
            if p == l:
                correct += 1
            if l_lbl != "O" and p_lbl != "O" and p == l:
                tp += 1
            elif l_lbl != "O" and (p_lbl == "O" or p != l):
                fn += 1
            elif l_lbl == "O" and p_lbl != "O":
                fp += 1

    acc = correct / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_qa_model(
    model_dir: str,
    test_data: List[Dict],
    output_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Load the trained joint Q&A model and evaluate on the test set.

    Args:
        model_dir:   Directory containing qa_model.pt.
        test_data:   List of dicts with keys: text, ner_labels, intent.
        output_path: Optional JSON output path.
        device:      Torch device.

    Returns:
        Combined NER + intent evaluation metrics.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(NER_CONFIG["bert_model"])

    model = JointQAModel(config=NER_CONFIG)
    trainer = QATrainer(model, config=NER_CONFIG, device=device)
    trainer.load(model_dir)

    label2id = model.label2id
    intent2id = model.intent2id

    test_dataset = QADataset(test_data, tokenizer, label2id, intent2id)
    test_loader = DataLoader(
        test_dataset, batch_size=NER_CONFIG["batch_size"], shuffle=False
    )

    metrics = trainer.evaluate(test_loader)

    print("\n[Joint Q&A Evaluation Results]")
    print("── NER (Design Knowledge Subject Extraction) ──")
    print(f"  Accuracy:  {metrics.get('ner_acc', 0)*100:.2f}%")
    print(f"  Precision: {metrics.get('ner_prec', 0)*100:.2f}%")
    print(f"  Recall:    {metrics.get('ner_rec', 0)*100:.2f}%")
    print(f"  F1:        {metrics.get('ner_f1', 0)*100:.2f}%")
    print()
    print("── Intent Classification (Query Intent) ──")
    print(f"  Accuracy:        {metrics.get('intent_acc', 0)*100:.2f}%")
    print(f"  Weighted Prec:   {metrics.get('intent_weighted_prec', 0)*100:.2f}%")
    print(f"  Weighted Recall: {metrics.get('intent_weighted_rec', 0)*100:.2f}%")
    print(f"  Weighted F1:     {metrics.get('intent_weighted_f1', 0)*100:.2f}%")
    print()
    print("[Paper Target (Table 10)]:")
    print("  NER:    Acc=97.78%, P=93.85%, R=92.12%, F1=92.97%")
    print("  Intent: Acc=85.32%, WP=87.37%, WR=86.12%, WF1=86.73%")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Model comparison table (Table 10)
# ---------------------------------------------------------------------------

def print_qa_comparison_table() -> None:
    """Print the Q&A model comparison from Table 10 of the paper."""
    data = {
        "BERT + CRF": {
            "ner_acc": 96.50, "ner_p": 86.46, "ner_r": 82.88, "ner_f1": 83.61,
            "intent_acc": None, "intent_wp": None, "intent_wr": None, "intent_wf1": None,
        },
        "BERT + Attention": {
            "ner_acc": None, "ner_p": None, "ner_r": None, "ner_f1": None,
            "intent_acc": 79.85, "intent_wp": 83.34, "intent_wr": 78.86, "intent_wf1": 81.03,
        },
        "BERT + CRF + Attention": {
            "ner_acc": 97.66, "ner_p": 91.25, "ner_r": 89.10, "ner_f1": 90.16,
            "intent_acc": 83.13, "intent_wp": 86.02, "intent_wr": 81.89, "intent_wf1": 83.89,
        },
        "Proposed ★": {
            "ner_acc": 97.78, "ner_p": 93.85, "ner_r": 92.12, "ner_f1": 92.97,
            "intent_acc": 85.32, "intent_wp": 87.37, "intent_wr": 86.12, "intent_wf1": 86.73,
        },
    }

    def fmt(v):
        return f"{v:.2f}" if v is not None else "  —  "

    print("\n[Table 10] Q&A Model Comparison (NER + Intent Classification)")
    print("-" * 100)
    header = (
        f"{'Model':<30} "
        f"{'NER Acc':>8} {'P':>7} {'R':>7} {'F1':>7}  |  "
        f"{'Int Acc':>8} {'WP':>7} {'WR':>7} {'WF1':>7}"
    )
    print(header)
    print("-" * 100)
    for model_name, scores in data.items():
        row = (
            f"{model_name:<30} "
            f"{fmt(scores['ner_acc']):>8} "
            f"{fmt(scores['ner_p']):>7} "
            f"{fmt(scores['ner_r']):>7} "
            f"{fmt(scores['ner_f1']):>7}  |  "
            f"{fmt(scores['intent_acc']):>8} "
            f"{fmt(scores['intent_wp']):>7} "
            f"{fmt(scores['intent_wr']):>7} "
            f"{fmt(scores['intent_wf1']):>7}"
        )
        print(row)
    print("-" * 100)
    print("★ = Proposed joint extraction model (LEBERT+BiLSTM-Attn-CRF + MultiScaleAttn)")


if __name__ == "__main__":
    print_qa_comparison_table()
