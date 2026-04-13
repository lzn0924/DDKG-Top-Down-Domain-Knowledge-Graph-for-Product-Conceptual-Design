"""
Joint extraction model for knowledge graph Q&A.

Encoder-decoder architecture:
  - Shared BERT encoder
  - Decoder 1: CRF for entity (subject) extraction
  - Decoder 2: Multi-scale attention for query intent classification
  - Adversarial training (FGM) for robustness
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    NER_CONFIG,
    NER_LABELS,
    INTENT_LABELS,
    NUM_INTENT_CLASSES,
    MODEL_DIR,
)
from knowledge_extraction.ner_model import CRF


# ---------------------------------------------------------------------------
# Multi-scale attention (query intent decoder)
# ---------------------------------------------------------------------------

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for query intent classification.

    Captures sentence information at different granularities:
      - Token-level (local):   single-token attention weights
      - Span-level (mid):      3-gram span attention
      - Sentence-level (global): global attention over the full sequence

    "The query intent decoder integrates a multi-scale attention mechanism,
    which accounts for sentence information at different granularities."
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Token-level (local) attention
        self.local_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        # Span-level: convolutional feature extraction
        self.span_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.span_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        # Sentence-level (global): mean pooling + attention
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fusion: combine three scales
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,      # (B, L, H)
        attention_mask: torch.Tensor,     # (B, L) – 1=real, 0=pad
    ) -> torch.Tensor:
        """Returns a (B, H) sentence representation."""
        key_padding_mask = ~attention_mask.bool()

        # Token-level attention (CLS as query)
        cls = hidden_states[:, :1, :]    # (B, 1, H)
        local_out, _ = self.local_attn(
            cls, hidden_states, hidden_states, key_padding_mask=key_padding_mask
        )
        local_out = local_out.squeeze(1)   # (B, H)

        # Span-level: 3-gram convolution → attention-pooling
        span_feat = self.span_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        span_out, _ = self.span_attn(
            cls, span_feat, span_feat, key_padding_mask=key_padding_mask
        )
        span_out = span_out.squeeze(1)     # (B, H)

        # Sentence-level: masked mean pooling
        mask_f = attention_mask.float().unsqueeze(-1)   # (B, L, 1)
        global_out = (hidden_states * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
        # (B, H)

        # Fuse three scales
        fused = torch.cat([local_out, span_out, global_out], dim=-1)  # (B, 3H)
        out = self.fusion(fused)    # (B, H)
        out = self.norm(self.dropout(out))
        return out


# ---------------------------------------------------------------------------
# Joint Q&A extraction model
# ---------------------------------------------------------------------------

class JointQAModel(nn.Module):
    """
    Shared BERT encoder + two decoders:
      Decoder 1 (CRF):                  Design knowledge subject extraction (NER)
      Decoder 2 (MultiScaleAttention):  Query intent classification (24 classes)

    Shared encoder with two task-specific decoders.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = config or NER_CONFIG

        self.num_ner_labels = len(NER_LABELS)
        self.num_intent_classes = NUM_INTENT_CLASSES
        self.label2id = {lbl: i for i, lbl in enumerate(NER_LABELS)}
        self.id2label = {i: lbl for i, lbl in enumerate(NER_LABELS)}
        self.intent2id = {intent: i for i, intent in enumerate(INTENT_LABELS)}
        self.id2intent = {i: intent for i, intent in enumerate(INTENT_LABELS)}

        # ── Shared encoder
        self.bert = AutoModel.from_pretrained(cfg["bert_model"])
        H = cfg["hidden_size"]
        self.dropout = nn.Dropout(cfg["dropout"])

        # ── Decoder 1: NER with CRF
        self.ner_fc = nn.Linear(H, self.num_ner_labels)
        self.crf = CRF(self.num_ner_labels)

        # ── Decoder 2: Multi-scale attention intent classifier
        self.intent_attn = MultiScaleAttention(
            hidden_size=H,
            num_heads=cfg["attention_heads"],
            dropout=cfg["dropout"],
        )
        self.intent_fc = nn.Linear(H, self.num_intent_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        ner_labels: Optional[torch.Tensor] = None,      # (B, L) BIO tags
        intent_labels: Optional[torch.Tensor] = None,   # (B,) intent class IDs
    ) -> Dict:
        # Shared BERT encoding
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state    # (B, L, H)
        bert_out = self.dropout(bert_out)

        # ── Decoder 1: NER
        ner_emissions = self.ner_fc(bert_out)    # (B, L, num_ner_labels)
        mask = attention_mask.bool()
        ner_preds = self.crf.decode(ner_emissions, mask=mask)

        # ── Decoder 2: Intent
        intent_repr = self.intent_attn(bert_out, attention_mask)   # (B, H)
        intent_logits = self.intent_fc(intent_repr)                 # (B, C)
        intent_preds = intent_logits.argmax(dim=-1)                 # (B,)

        result = {
            "ner_emissions": ner_emissions,
            "ner_predictions": ner_preds,
            "intent_logits": intent_logits,
            "intent_predictions": intent_preds,
        }

        # ── Losses
        if ner_labels is not None or intent_labels is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)
            w_ner = NER_CONFIG["task_loss_weight_ner"]          # 0.75
            w_intent = NER_CONFIG["adversarial_loss_weight_intent"]  # 0.25

            if ner_labels is not None:
                ner_loss = self.crf(
                    ner_emissions, ner_labels, mask=mask, reduction="mean"
                )
                total_loss = total_loss + w_ner * ner_loss
                result["ner_loss"] = ner_loss

            if intent_labels is not None:
                intent_loss = F.cross_entropy(intent_logits, intent_labels)
                total_loss = total_loss + w_intent * intent_loss
                result["intent_loss"] = intent_loss

            result["loss"] = total_loss

        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QADataset(Dataset):
    """
    Dataset for joint NER + intent classification training.

    Each sample:
      - text:         Query string (Chinese)
      - ner_labels:   Character-level BIO label sequence
      - intent_label: Query intent class ID
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        intent2id: Dict[str, int],
        max_length: int = NER_CONFIG["max_seq_len"],
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.intent2id = intent2id
        self.max_length = max_length
        self.samples = self._encode(samples)

    def _encode(self, samples: List[Dict]) -> List[Dict]:
        encoded = []
        for sample in samples:
            text = sample["text"]
            ner_labels = sample.get("ner_labels", ["O"] * len(text))
            intent = sample.get("intent", "design_style")

            encoding = self.tokenizer(
                list(text),
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            label_ids = [0] + [
                self.label2id.get(lbl, 0) for lbl in ner_labels
            ][:self.max_length - 2] + [0]
            label_ids += [0] * (self.max_length - len(label_ids))
            label_ids = label_ids[:self.max_length]

            encoded.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "token_type_ids": encoding.get(
                    "token_type_ids",
                    torch.zeros_like(encoding["input_ids"])
                ).squeeze(0),
                "ner_labels": torch.tensor(label_ids, dtype=torch.long),
                "intent_labels": torch.tensor(
                    self.intent2id.get(intent, 0), dtype=torch.long
                ),
            })
        return encoded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Adversarial training (FGM – Fast Gradient Method)
# ---------------------------------------------------------------------------

class FGMAdversarialTrainer:
    """
    Fast Gradient Method adversarial training.

    Perturbs word embedding gradients to generate virtual adversarial examples,
    improving model robustness to input variations.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        self._backup: Dict[str, torch.Tensor] = {}

    def attack(self, emb_name: str = "word_embeddings") -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self._backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self, emb_name: str = "word_embeddings") -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class QATrainer:
    """Training loop for the joint Q&A extraction model."""

    def __init__(
        self,
        model: JointQAModel,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.cfg = config or NER_CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fgm = FGMAdversarialTrainer(self.model)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> List[Dict]:
        num_epochs = num_epochs or self.cfg["num_epochs"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=0.01,
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        history = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]

                optimizer.zero_grad()
                loss.backward()

                # Adversarial training: generate adversarial perturbation and update
                n_adv = self.cfg["num_adversarial_per_sample"]
                for _ in range(n_adv):
                    self.fgm.attack()
                    adv_outputs = self.model(**batch)
                    adv_loss = adv_outputs["loss"] / n_adv
                    adv_loss.backward()
                    self.fgm.restore()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            metrics = {"epoch": epoch + 1, "train_loss": avg_loss}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics.update(val_metrics)

            history.append(metrics)
            print(f"[QA] Epoch {epoch+1}/{num_epochs} – " +
                  " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items()))
        return history

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict:
        self.model.eval()
        ner_preds_all, ner_labels_all = [], []
        intent_preds_all, intent_labels_all = [], []

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            ner_preds = outputs["ner_predictions"]
            ner_labels = batch["ner_labels"].cpu().tolist()
            mask = batch["attention_mask"].cpu().tolist()
            for pred, lbl, msk in zip(ner_preds, ner_labels, mask):
                seq_len = sum(msk)
                ner_preds_all.append(pred[:seq_len])
                ner_labels_all.append(lbl[:seq_len])

            intent_preds_all.extend(outputs["intent_predictions"].cpu().tolist())
            intent_labels_all.extend(batch["intent_labels"].cpu().tolist())

        ner_metrics = self._compute_ner_metrics(ner_preds_all, ner_labels_all)
        intent_metrics = self._compute_intent_metrics(intent_preds_all, intent_labels_all)
        return {**ner_metrics, **intent_metrics}

    def _compute_ner_metrics(
        self,
        predictions: List[List[int]],
        labels: List[List[int]],
    ) -> Dict[str, float]:
        tp = fp = fn = 0
        total = correct = 0
        for pred_seq, lbl_seq in zip(predictions, labels):
            for p, l in zip(pred_seq, lbl_seq):
                total += 1
                if p == l:
                    correct += 1
                p_lbl = self.model.id2label.get(p, "O")
                l_lbl = self.model.id2label.get(l, "O")
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
        return {"ner_acc": acc, "ner_prec": prec, "ner_rec": rec, "ner_f1": f1}

    def _compute_intent_metrics(
        self,
        predictions: List[int],
        labels: List[int],
    ) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        if not predictions:
            return {}
        acc = accuracy_score(labels, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        return {
            "intent_acc": acc,
            "intent_weighted_prec": prec,
            "intent_weighted_rec": rec,
            "intent_weighted_f1": f1,
        }

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "qa_model.pt"))
        print(f"[QA] Model saved to {save_dir}")

    def load(self, save_dir: str) -> None:
        path = os.path.join(save_dir, "qa_model.pt")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[QA] Model loaded from {path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, text: str, tokenizer) -> Dict:
        """
        Run inference on a single query string.

        Returns:
            Dict with keys:
              entities: list of (entity_text, label) pairs
              intent:   predicted intent label string
        """
        self.model.eval()
        encoding = tokenizer(
            list(text),
            is_split_into_words=True,
            max_length=NER_CONFIG["max_seq_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Decode NER
        pred_ids = outputs["ner_predictions"][0]
        real_len = batch["attention_mask"].sum().item()
        pred_labels = [self.model.id2label.get(i, "O") for i in pred_ids[:real_len]]

        entities = []
        current_entity = []
        current_label = None
        for char, lbl in zip(text, pred_labels[1:real_len - 1]):
            if lbl.startswith("B-"):
                if current_entity:
                    entities.append(("".join(current_entity), current_label))
                current_entity = [char]
                current_label = lbl[2:]
            elif lbl.startswith("I-") and current_entity:
                current_entity.append(char)
            else:
                if current_entity:
                    entities.append(("".join(current_entity), current_label))
                    current_entity = []
                    current_label = None

        if current_entity:
            entities.append(("".join(current_entity), current_label))

        # Decode intent
        intent_id = outputs["intent_predictions"][0].item()
        intent = self.model.id2intent.get(intent_id, "unknown")

        return {"entities": entities, "intent": intent}
