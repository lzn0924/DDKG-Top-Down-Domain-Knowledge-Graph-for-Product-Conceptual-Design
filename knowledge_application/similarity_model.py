"""
Unsupervised contrastive learning model for semantic similarity.

SimCSE-style approach:
  - Backbone: BERT-WWM (hfl/chinese-bert-wwm-ext)
  - Positive pairs: same sentence, two different dropout masks
  - In-batch negatives: all other sentences in the batch
  - NT-Xent loss with temperature τ

Pooling strategies (P1–P4):
  P1: CLS token (last layer)
  P2: NSP pooler output
  P3: Mean pooling (last layer)
  P4: Mean pooling (first + last layers)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import SIMILARITY_CONFIG, MODEL_DIR


# ---------------------------------------------------------------------------
# Pooling strategies (P1 – P4)
# ---------------------------------------------------------------------------

def pool_output(
    bert_output,
    attention_mask: torch.Tensor,
    strategy: str = "P4",
) -> torch.Tensor:
    """
    Apply one of the four pooling strategies (P1–P4).

    Args:
        bert_output:    HuggingFace BaseModelOutput with last_hidden_state,
                        pooler_output, and hidden_states.
        attention_mask: (B, L) boolean mask.
        strategy:       'P1' | 'P2' | 'P3' | 'P4'

    Returns:
        Sentence embedding tensor of shape (B, H).
    """
    if strategy == "P1":
        # CLS vector of last encoder layer
        return bert_output.last_hidden_state[:, 0, :]

    elif strategy == "P2":
        # BERT NSP pooler vector
        return bert_output.pooler_output

    elif strategy == "P3":
        # Mean of all token vectors in last layer (masked)
        mask = attention_mask.float().unsqueeze(-1)   # (B, L, 1)
        hidden = bert_output.last_hidden_state          # (B, L, H)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    elif strategy == "P4":
        # Mean of all vectors in first and last layers
        # Requires output_hidden_states=True
        first = bert_output.hidden_states[1]    # First transformer layer output
        last = bert_output.last_hidden_state    # Last layer
        combined = (first + last) / 2.0
        mask = attention_mask.float().unsqueeze(-1)
        return (combined * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent)
# ---------------------------------------------------------------------------

def contrastive_loss(
    embeddings_a: torch.Tensor,   # (B, H)
    embeddings_b: torch.Tensor,   # (B, H) – positive pairs for each in A
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    For each anchor i in batch A:
      - Positive:  embeddings_b[i]  (same sentence, different dropout)
      - Negatives: all other sentences in the batch (in-batch negatives)

    Loss = -log( exp(sim(i, pos_i) / τ) / Σ_j exp(sim(i, j) / τ) )

    Args:
        embeddings_a: Anchor sentence embeddings.
        embeddings_b: Positive pair embeddings.
        temperature:  Temperature τ (default 0.05).

    Returns:
        Scalar loss.
    """
    # L2 normalize
    a = F.normalize(embeddings_a, dim=-1)   # (B, H)
    b = F.normalize(embeddings_b, dim=-1)   # (B, H)

    # Similarity matrix: (B, 2B) – compare each anchor against all positives + negatives
    all_embeddings = torch.cat([a, b], dim=0)   # (2B, H)
    sim_matrix = torch.mm(a, all_embeddings.T) / temperature   # (B, 2B)

    # Labels: for anchor i, the positive is at position B + i in all_embeddings
    batch_size = a.size(0)
    labels = torch.arange(batch_size, device=a.device) + batch_size   # (B,)

    # Mask out self-similarity (diagonal of A-vs-A block)
    mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    for i in range(batch_size):
        mask[i, i] = True
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ContrastiveSimilarityModel(nn.Module):
    """
    Unsupervised contrastive learning model for domain-specific semantic similarity.

    Domain knowledge fusion:
      - Train on generic + domain sentences combined
      - Dropout augmentation generates positive pairs automatically
      - Temperature τ = 0.05 for contrastive objective
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = config or SIMILARITY_CONFIG
        self.cfg = cfg
        self.pooling = cfg.get("pooling", "P4")

        # BERT-WWM backbone (output_hidden_states=True required for P4)
        self.bert = AutoModel.from_pretrained(
            cfg["bert_model"],
            output_hidden_states=True,
        )
        self.dropout = nn.Dropout(cfg["dropout"])

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input tokens to a sentence embedding vector."""
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return pool_output(output, attention_mask, strategy=self.pooling)

    def forward(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        token_type_ids_a: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass.

        Training mode (unsupervised):
          - input_ids_b is None → create positive pair by passing input_ids_a
            through the encoder a second time with a different dropout mask.

        Inference mode:
          - input_ids_b provided → compute cosine similarity between A and B.
        """
        emb_a = self.encode(input_ids_a, attention_mask_a, token_type_ids_a)

        if input_ids_b is None:
            # Unsupervised: second pass with dropout for positive pair
            emb_b = self.encode(input_ids_a, attention_mask_a, token_type_ids_a)
        else:
            emb_b = self.encode(input_ids_b, attention_mask_b, token_type_ids_b)

        result = {"embeddings_a": emb_a, "embeddings_b": emb_b}

        if self.training:
            loss = contrastive_loss(emb_a, emb_b, temperature=self.cfg["temperature"])
            result["loss"] = loss
        else:
            # Cosine similarity for inference/evaluation
            sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
            result["similarity"] = sim

        return result


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class UnsupervisedSimilarityDataset(Dataset):
    """
    Dataset for unsupervised contrastive training.
    Each sample is a single sentence; the positive pair is generated on-the-fly
    by passing the same sentence through the encoder twice with different dropout.

    Mixes:
      - 5000 generic sentences (Chinese-SNLI + STS-B)
      - 500 home design domain sentences
    """

    def __init__(
        self,
        sentences: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids_a": encoding["input_ids"].squeeze(0),
            "attention_mask_a": encoding["attention_mask"].squeeze(0),
        }


class STSDataset(Dataset):
    """
    Dataset for evaluation on STS-B format pairs.

    Format (STS-B style):
      Each: (sentence1, sentence2, similarity_score ∈ [0, 5])
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        enc_a = self.tokenizer(
            pair["sentence1"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            pair["sentence2"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "score": torch.tensor(pair["score"], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SimilarityTrainer:
    """
    Training and evaluation for the contrastive similarity model.

    Supports ablation across P1–P4 pooling strategies and different BERT backbones.
    """

    def __init__(
        self,
        model: ContrastiveSimilarityModel,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.cfg = config or SIMILARITY_CONFIG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> List[Dict]:
        """Train the contrastive model."""
        num_epochs = num_epochs or self.cfg["num_epochs"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg["learning_rate"],
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            metrics = {"epoch": epoch + 1, "train_loss": avg_loss}

            if val_loader is not None:
                spearman = self.evaluate_spearman(val_loader)
                metrics["val_spearman"] = spearman
                print(
                    f"[Similarity] Epoch {epoch+1}/{num_epochs} – "
                    f"loss={avg_loss:.4f} | val_spearman={spearman:.4f}"
                )
            else:
                print(f"[Similarity] Epoch {epoch+1}/{num_epochs} – loss={avg_loss:.4f}")

            history.append(metrics)
        return history

    @torch.no_grad()
    def evaluate_spearman(self, data_loader: DataLoader) -> float:
        """
        Compute Spearman's rank correlation on STS pairs.

        Primary evaluation metric: Spearman's ρ between predicted and gold scores.
        """
        self.model.eval()
        all_preds, all_gold = [], []

        for batch in data_loader:
            scores = batch.pop("score").numpy()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            sims = outputs["similarity"].cpu().numpy()
            all_preds.extend(sims.tolist())
            all_gold.extend(scores.tolist())

        rho, _ = spearmanr(all_gold, all_preds)
        return float(rho)

    @torch.no_grad()
    def encode_sentences(self, sentences: List[str], tokenizer, batch_size: int = 32) -> np.ndarray:
        """Encode a list of sentences to embedding vectors for downstream use."""
        self.model.eval()
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i: i + batch_size]
            encoding = tokenizer(
                batch_sents,
                max_length=self.cfg["max_seq_len"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            output = self.model.bert(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            # Use configured pooling strategy
            emb = pool_output(output, encoding["attention_mask"], self.model.pooling)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "similarity_model.pt"))
        print(f"[Similarity] Model saved to {save_dir}")

    def load(self, save_dir: str) -> None:
        path = os.path.join(save_dir, "similarity_model.pt")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[Similarity] Model loaded from {path}")


# ---------------------------------------------------------------------------
# Ablation: pooling strategy comparison
# ---------------------------------------------------------------------------

def ablation_pooling_strategies(
    model_name: str,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all four pooling strategies on the same BERT backbone.
    Evaluates P1–P4 strategies on the given backbone.

    Args:
        model_name:  HuggingFace model identifier.
        val_loader:  STS validation DataLoader.
        test_loader: STS test DataLoader.
        device:      Torch device string.

    Returns:
        {strategy: {"val_spearman": float, "test_spearman": float}}
    """
    results = {}
    for strategy in ["P1", "P2", "P3", "P4"]:
        cfg = dict(SIMILARITY_CONFIG)
        cfg["bert_model"] = model_name
        cfg["pooling"] = strategy

        model = ContrastiveSimilarityModel(config=cfg)
        # Evaluate without contrastive training (zero-shot pooling baseline)
        trainer = SimilarityTrainer(model, config=cfg, device=device)
        val_rho = trainer.evaluate_spearman(val_loader)
        test_rho = trainer.evaluate_spearman(test_loader)
        results[f"{model_name.split('/')[-1]}-{strategy}"] = {
            "val_spearman": val_rho,
            "test_spearman": test_rho,
        }
        print(f"  {strategy}: val={val_rho:.4f} | test={test_rho:.4f}")
    return results
