"""
Incremental annotation mechanism for semi-supervised knowledge base construction.

Four-step process:
  (1) Automatic Extraction – TF-IDF + word embeddings identify candidate terms
  (2) Batched Expert Review  – candidates batched and sent to domain experts
  (3) Expert Annotation      – experts annotate relevance; provide modifications
  (4) Knowledge Base Update  – accepted terms merged into ontology KB
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from config import DATA_DIR, CLUSTERING_CONFIG


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TermCandidate:
    """A candidate term extracted automatically from the corpus."""
    term: str
    tfidf_score: float
    embedding_similarity: float   # Similarity to nearest seed tag
    nearest_tag: str
    source_documents: List[str] = field(default_factory=list)
    # Annotation fields (filled by experts)
    annotation_status: str = "pending"  # pending | accepted | rejected | modified
    annotated_label: Optional[str] = None
    expert_note: Optional[str] = None


@dataclass
class AnnotationBatch:
    """A batch of candidates sent to experts for annotation."""
    batch_id: str
    created_at: float
    candidates: List[TermCandidate]
    annotator: Optional[str] = None
    completed_at: Optional[float] = None
    status: str = "open"   # open | completed


@dataclass
class KnowledgeBase:
    """The cumulative domain terminology knowledge base."""
    terms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # terms[term_string] = {tag, score, source, iteration}
    iteration: int = 0
    total_auto_extracted: int = 0
    total_manual: int = 0

    def add_term(self, term: str, tag: str, score: float,
                 source: str = "auto", iteration: int = 0) -> None:
        if term not in self.terms:
            self.terms[term] = {
                "tag": tag,
                "score": score,
                "source": source,
                "iteration": iteration,
            }

    @property
    def manual_ratio(self) -> float:
        total = len(self.terms)
        if total == 0:
            return 0.0
        manual = sum(1 for v in self.terms.values() if v["source"] == "manual")
        return manual / total

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kb = cls()
        kb.terms = data.get("terms", {})
        kb.iteration = data.get("iteration", 0)
        kb.total_auto_extracted = data.get("total_auto_extracted", 0)
        kb.total_manual = data.get("total_manual", 0)
        return kb


# ---------------------------------------------------------------------------
# Incremental annotation engine
# ---------------------------------------------------------------------------

class IncrementalAnnotationEngine:
    """
    Orchestrates the incremental annotation cycle.

    Cycle (repeated until convergence or max_iterations):
      1. Run TF-IDF + embedding extraction on new corpus slices
      2. Filter unmatched phrases → candidates
      3. Batch candidates → expert review queue
      4. Integrate accepted/modified terms into knowledge base
      5. Re-train TF-IDF/embedding models with updated KB

    This design significantly reduces the manual annotation workload.
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        batch_size: int = 50,
        max_iterations: int = 10,
        auto_accept_threshold: float = 0.90,  # High-confidence auto-accept
        review_threshold: float = 0.70,        # Send to expert if [review_thr, auto_thr)
        reject_threshold: float = 0.70,        # Auto-reject if below reject_thr
        kb_save_path: Optional[str] = None,
    ):
        self.kb = knowledge_base or KnowledgeBase()
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold
        self.reject_threshold = reject_threshold
        self.kb_save_path = kb_save_path or os.path.join(
            DATA_DIR, "knowledge_base.json"
        )
        self._batches: List[AnnotationBatch] = []
        self._batch_counter = 0

    # ------------------------------------------------------------------
    # Step 1: Automatic extraction (deferred to TagBasedClusteringPipeline)
    # ------------------------------------------------------------------

    def filter_candidates(
        self,
        raw_clusters: Dict[str, List[str]],
        scores: Dict[str, float],
    ) -> Tuple[List[TermCandidate], List[TermCandidate]]:
        """
        Partition extracted candidates into:
          - auto_accepted: high confidence, add directly to KB
          - review_queue:  medium confidence, send to experts

        Args:
            raw_clusters: {tag: [terms...]} from TagBasedClusteringPipeline
            scores:       {term: similarity_score}

        Returns:
            (auto_accepted_list, review_queue_list)
        """
        auto_accepted: List[TermCandidate] = []
        review_queue: List[TermCandidate] = []

        for tag, terms in raw_clusters.items():
            for term in terms:
                score = scores.get(term, 0.0)
                candidate = TermCandidate(
                    term=term,
                    tfidf_score=score,
                    embedding_similarity=score,
                    nearest_tag=tag,
                )
                if term in self.kb.terms:
                    continue   # Already in KB
                if score >= self.auto_accept_threshold:
                    candidate.annotation_status = "accepted"
                    auto_accepted.append(candidate)
                elif score >= self.review_threshold:
                    review_queue.append(candidate)
                # else: silently reject low-confidence candidates

        return auto_accepted, review_queue

    # ------------------------------------------------------------------
    # Step 2: Batched expert review
    # ------------------------------------------------------------------

    def create_batch(self, candidates: List[TermCandidate]) -> List[AnnotationBatch]:
        """Split candidates into fixed-size batches for expert review."""
        batches = []
        for i in range(0, len(candidates), self.batch_size):
            self._batch_counter += 1
            batch = AnnotationBatch(
                batch_id=f"batch_{self._batch_counter:04d}",
                created_at=time.time(),
                candidates=candidates[i: i + self.batch_size],
            )
            batches.append(batch)
            self._batches.append(batch)
        return batches

    def export_batch_for_review(self, batch: AnnotationBatch, output_dir: str) -> str:
        """Export a batch to a JSON file for expert review."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{batch.batch_id}.json")
        export_data = {
            "batch_id": batch.batch_id,
            "created_at": batch.created_at,
            "candidates": [
                {
                    "term": c.term,
                    "suggested_tag": c.nearest_tag,
                    "confidence": round(c.embedding_similarity, 4),
                    "annotation": "pending",    # Expert fills: accept | reject | modified_term
                    "note": "",
                }
                for c in batch.candidates
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        return path

    def import_annotated_batch(self, path: str) -> AnnotationBatch:
        """Import an expert-annotated batch from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find corresponding batch object
        batch_id = data["batch_id"]
        batch = next((b for b in self._batches if b.batch_id == batch_id), None)
        if batch is None:
            raise ValueError(f"Batch {batch_id} not found.")

        # Apply annotations
        for annot, cand in zip(data["candidates"], batch.candidates):
            status = annot.get("annotation", "pending")
            if status == "accept":
                cand.annotation_status = "accepted"
            elif status == "reject":
                cand.annotation_status = "rejected"
            elif status.startswith("modify:"):
                cand.annotation_status = "modified"
                cand.annotated_label = status.split("modify:", 1)[1].strip()
            cand.expert_note = annot.get("note", "")

        batch.completed_at = time.time()
        batch.status = "completed"
        return batch

    # ------------------------------------------------------------------
    # Step 3: Knowledge base update
    # ------------------------------------------------------------------

    def update_knowledge_base(
        self,
        auto_accepted: List[TermCandidate],
        annotated_batches: List[AnnotationBatch],
    ) -> None:
        """
        Integrate accepted and modified candidates into the knowledge base.

        Auto-accepted candidates are added with source='auto'.
        Expert-confirmed candidates are added with source='expert'.
        """
        self.kb.iteration += 1

        for cand in auto_accepted:
            self.kb.add_term(
                term=cand.term,
                tag=cand.nearest_tag,
                score=cand.embedding_similarity,
                source="auto",
                iteration=self.kb.iteration,
            )
            self.kb.total_auto_extracted += 1

        for batch in annotated_batches:
            for cand in batch.candidates:
                if cand.annotation_status == "accepted":
                    self.kb.add_term(
                        term=cand.term,
                        tag=cand.nearest_tag,
                        score=cand.embedding_similarity,
                        source="expert",
                        iteration=self.kb.iteration,
                    )
                    self.kb.total_manual += 1
                elif cand.annotation_status == "modified":
                    new_term = cand.annotated_label or cand.term
                    self.kb.add_term(
                        term=new_term,
                        tag=cand.nearest_tag,
                        score=cand.embedding_similarity,
                        source="expert_modified",
                        iteration=self.kb.iteration,
                    )
                    self.kb.total_manual += 1

        self.kb.save(self.kb_save_path)
        print(
            f"[IncrementalAnnotation] Iteration {self.kb.iteration}: "
            f"KB size={len(self.kb.terms)}, "
            f"manual%={self.kb.manual_ratio * 100:.2f}%"
        )

    # ------------------------------------------------------------------
    # Convenience: simulate expert annotation (for testing/ablation)
    # ------------------------------------------------------------------

    def simulate_expert_annotation(
        self,
        batch: AnnotationBatch,
        accept_rate: float = 0.85,
        modify_rate: float = 0.10,
    ) -> None:
        """
        Simulate expert annotation for testing purposes.
        In the real system, this step is performed by human domain experts.

        accept_rate + modify_rate + reject_rate must sum to ≤ 1.
        """
        import random
        rng = random.Random(42)
        for cand in batch.candidates:
            r = rng.random()
            if r < accept_rate:
                cand.annotation_status = "accepted"
            elif r < accept_rate + modify_rate:
                cand.annotation_status = "modified"
                cand.annotated_label = cand.term + "_精确"
            else:
                cand.annotation_status = "rejected"
        batch.status = "completed"
        batch.completed_at = time.time()

    # ------------------------------------------------------------------
    # Full iterative loop
    # ------------------------------------------------------------------

    def run_iteration(
        self,
        raw_clusters: Dict[str, List[str]],
        term_scores: Dict[str, float],
        simulate: bool = False,
        review_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute one full annotation iteration.

        Args:
            raw_clusters:      Clusters from TagBasedClusteringPipeline.
            term_scores:       Per-term TF-IDF/similarity scores.
            simulate:          If True, auto-simulate expert annotation.
            review_output_dir: Directory to export batch files.

        Returns:
            Summary statistics for this iteration.
        """
        auto_accepted, review_queue = self.filter_candidates(raw_clusters, term_scores)
        print(
            f"[IncrementalAnnotation] Auto-accepted={len(auto_accepted)}, "
            f"Review queue={len(review_queue)}"
        )

        batches = self.create_batch(review_queue)
        annotated_batches = []
        for batch in batches:
            if simulate:
                self.simulate_expert_annotation(batch)
            else:
                # In production: export → wait for expert → import
                if review_output_dir:
                    path = self.export_batch_for_review(
                        batch, review_output_dir
                    )
                    print(f"[IncrementalAnnotation] Exported batch to {path}")
            annotated_batches.append(batch)

        self.update_knowledge_base(auto_accepted, annotated_batches)

        return {
            "iteration": self.kb.iteration,
            "kb_size": len(self.kb.terms),
            "manual_ratio": self.kb.manual_ratio,
            "new_auto": len(auto_accepted),
            "new_manual": sum(
                1 for b in annotated_batches
                for c in b.candidates
                if c.annotation_status in ("accepted", "modified")
            ),
        }
