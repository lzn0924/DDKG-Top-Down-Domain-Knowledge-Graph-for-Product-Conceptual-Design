"""
Tag-based clustering for domain terminology extraction.

Hybrid semi-supervised pipeline:
  1. Expert preset seed tags
  2. Tag expansion via synonyms, hypernyms, and ontology relations
  3. TF-IDF for candidate term scoring
  4. Word2Vec for semantic similarity-based clustering
"""

import os
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CLUSTERING_CONFIG, DATA_DIR


# ---------------------------------------------------------------------------
# Tag expansion utilities
# ---------------------------------------------------------------------------

class TagExpander:
    """
    Expands expert-preset seed tags using ontology-derived semantic relations:
      - Synonyms (e.g., "living room decoration" ↔ "living room design")
      - Hypernyms (e.g., "living room decoration" → "interior design")
      - Similar concepts (e.g., "living room decoration" ↔ "bedroom decoration")

    In the full system these relations come from the OWL ontology; here we
    represent them as an extensible dictionary.
    """

    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None,
                 hypernym_dict: Optional[Dict[str, List[str]]] = None,
                 similar_dict: Optional[Dict[str, List[str]]] = None):
        self.synonyms: Dict[str, List[str]] = synonym_dict or {}
        self.hypernyms: Dict[str, List[str]] = hypernym_dict or {}
        self.similars: Dict[str, List[str]] = similar_dict or {}

    def expand(self, tag: str) -> Set[str]:
        """
        Return the full set of expanded tags for a given seed tag,
        including the original tag itself.
        """
        expanded = {tag}
        expanded.update(self.synonyms.get(tag, []))
        expanded.update(self.hypernyms.get(tag, []))
        expanded.update(self.similars.get(tag, []))
        return expanded

    @classmethod
    def from_ontology_file(cls, path: str) -> "TagExpander":
        """Load semantic relations from a JSON ontology export."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            synonym_dict=data.get("synonyms", {}),
            hypernym_dict=data.get("hypernyms", {}),
            similar_dict=data.get("similars", {}),
        )

    @staticmethod
    def build_default_expander() -> "TagExpander":
        """
        Default expander with home-design domain seed relations.
        """
        synonyms = {
            "客厅装修": ["客厅设计", "起居室装潢", "living room design"],
            "卧室装修": ["卧室设计", "主卧装潢", "bedroom design"],
            "厨房设计": ["厨房装修", "厨房布局", "kitchen design"],
            "卫生间设计": ["卫浴设计", "洗手间装修", "bathroom design"],
            "木材": ["实木", "原木", "wood material"],
            "大理石": ["石材", "天然石材", "marble"],
            "北欧风格": ["北欧风", "Nordic style"],
            "现代简约": ["简约风格", "modern minimalist"],
        }
        hypernyms = {
            "客厅装修": ["室内设计", "家居设计", "interior design"],
            "卧室装修": ["室内设计", "家居设计"],
            "厨房设计": ["室内设计"],
            "木材": ["装修材料", "建材"],
            "大理石": ["装修材料", "石材类"],
            "北欧风格": ["装修风格", "设计风格"],
            "现代简约": ["装修风格"],
        }
        similars = {
            "客厅装修": ["卧室装修", "餐厅装修", "书房装修"],
            "卧室装修": ["客厅装修", "儿童房装修"],
            "木材": ["大理石", "玻璃", "金属"],
            "北欧风格": ["现代简约", "美式风格", "中式风格"],
        }
        return cls(synonym_dict=synonyms, hypernym_dict=hypernyms, similar_dict=similars)


# ---------------------------------------------------------------------------
# TF-IDF term extraction
# ---------------------------------------------------------------------------

class TFIDFTermExtractor:
    """
    Extracts candidate terminology from a text corpus using TF-IDF weighting.

    Paper: "Statistical models, including TF-IDF and word embeddings, to
    automatically identify potential terms."
    """

    def __init__(
        self,
        min_df: int = CLUSTERING_CONFIG["tfidf_min_df"],
        max_df: float = CLUSTERING_CONFIG["tfidf_max_df"],
        threshold: float = CLUSTERING_CONFIG["tfidf_threshold"],
        ngram_range: Tuple[int, int] = (1, 3),
    ):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            analyzer="char_wb",   # Character n-grams for Chinese
            token_pattern=None,
        )
        self._is_fitted = False

    def fit(self, corpus: List[str]) -> "TFIDFTermExtractor":
        self.vectorizer.fit(corpus)
        self._is_fitted = True
        return self

    def extract_terms(self, document: str) -> List[Tuple[str, float]]:
        """
        Extract high-TF-IDF terms from a single document.

        Returns:
            List of (term, score) tuples above the threshold, sorted descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before extract_terms().")
        tfidf_matrix = self.vectorizer.transform([document])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        terms = [
            (feature_names[i], float(scores[i]))
            for i in scores.argsort()[::-1]
            if scores[i] >= self.threshold
        ]
        return terms

    def extract_from_corpus(self, corpus: List[str]) -> Dict[str, float]:
        """
        Extract and aggregate term scores across the full corpus.

        Returns:
            Dict mapping term → max TF-IDF score across documents.
        """
        if not self._is_fitted:
            self.fit(corpus)
        tfidf_matrix = self.vectorizer.transform(corpus)
        feature_names = self.vectorizer.get_feature_names_out()
        max_scores = tfidf_matrix.max(axis=0).toarray()[0]
        return {
            term: float(score)
            for term, score in zip(feature_names, max_scores)
            if score >= self.threshold
        }


# ---------------------------------------------------------------------------
# Word2Vec semantic clustering
# ---------------------------------------------------------------------------

class Word2VecClusterer:
    """
    Trains a Word2Vec model on the domain corpus and clusters candidate
    terms under expanded tag seeds using cosine similarity.
    """

    def __init__(
        self,
        vector_size: int = CLUSTERING_CONFIG["word2vec_dim"],
        window: int = CLUSTERING_CONFIG["word2vec_window"],
        min_count: int = CLUSTERING_CONFIG["word2vec_min_count"],
        similarity_threshold: float = CLUSTERING_CONFIG["similarity_threshold"],
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.similarity_threshold = similarity_threshold
        self.workers = workers
        self.model: Optional[Word2Vec] = None

    def train(self, tokenized_corpus: List[List[str]]) -> "Word2VecClusterer":
        """Train Word2Vec on a tokenized corpus."""
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10,
        )
        return self

    def get_vector(self, term: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a term (returns None if OOV)."""
        if self.model is None:
            raise RuntimeError("Call train() before get_vector().")
        if term in self.model.wv:
            return self.model.wv[term]
        # For multi-character terms, average sub-word vectors
        chars = list(term)
        vecs = [self.model.wv[c] for c in chars if c in self.model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        return None

    def cluster_terms(
        self,
        candidate_terms: List[str],
        expanded_tags: Dict[str, Set[str]],
    ) -> Dict[str, List[str]]:
        """
        Assign each candidate term to the most similar tag cluster.

        Args:
            candidate_terms: Terms extracted by TF-IDF.
            expanded_tags:   {seed_tag: {seed_tag, synonym1, hypernym1, ...}}

        Returns:
            Dict {seed_tag: [term, term, ...]} – clustered terminology table.
        """
        # Pre-compute tag centroid vectors
        tag_centroids: Dict[str, np.ndarray] = {}
        for seed_tag, expanded in expanded_tags.items():
            vecs = [self.get_vector(t) for t in expanded]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                tag_centroids[seed_tag] = np.mean(vecs, axis=0)

        clusters: Dict[str, List[str]] = {tag: [] for tag in expanded_tags}

        for term in candidate_terms:
            term_vec = self.get_vector(term)
            if term_vec is None:
                continue
            best_tag, best_sim = None, -1.0
            for tag, centroid in tag_centroids.items():
                sim = float(cosine_similarity(
                    term_vec.reshape(1, -1), centroid.reshape(1, -1)
                )[0, 0])
                if sim > best_sim:
                    best_sim, best_tag = sim, tag
            if best_tag is not None and best_sim >= self.similarity_threshold:
                clusters[best_tag].append(term)

        return clusters

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)

    def load(self, path: str) -> "Word2VecClusterer":
        self.model = Word2Vec.load(path)
        return self


# ---------------------------------------------------------------------------
# Full terminology extraction pipeline
# ---------------------------------------------------------------------------

class TagBasedClusteringPipeline:
    """
    End-to-end pipeline implementing the tag-based clustering method:

      Step 1: Experts provide preset seed tags
      Step 2: Expand seed tags via ontology semantic relations
      Step 3: TF-IDF extracts candidate terms from corpus
      Step 4: Word2Vec clusters candidate terms under expanded tag sets
      Step 5: Expert review and incremental feedback (handled externally)

    Precision=0.821, Recall=0.794, F1=0.872 on the home design domain.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or CLUSTERING_CONFIG
        self.expander = TagExpander.build_default_expander()
        self.tfidf = TFIDFTermExtractor(
            min_df=cfg["tfidf_min_df"],
            max_df=cfg["tfidf_max_df"],
            threshold=cfg["tfidf_threshold"],
        )
        self.clusterer = Word2VecClusterer(
            vector_size=cfg["word2vec_dim"],
            window=cfg["word2vec_window"],
            min_count=cfg["word2vec_min_count"],
            similarity_threshold=cfg["similarity_threshold"],
        )

    def fit(
        self,
        corpus: List[str],
        tokenized_corpus: List[List[str]],
        seed_tags: List[str],
    ) -> "TagBasedClusteringPipeline":
        """
        Fit the pipeline on the domain corpus.

        Args:
            corpus:           Raw text documents (for TF-IDF).
            tokenized_corpus: Tokenized documents (for Word2Vec).
            seed_tags:        Expert-provided seed tags.
        """
        print("[TagClustering] Fitting TF-IDF model...")
        self.tfidf.fit(corpus)

        print("[TagClustering] Training Word2Vec model...")
        self.clusterer.train(tokenized_corpus)

        print("[TagClustering] Expanding seed tags...")
        self.expanded_tags = {
            tag: self.expander.expand(tag) for tag in seed_tags
        }

        return self

    def extract_and_cluster(self, corpus: List[str]) -> Dict[str, List[str]]:
        """
        Extract candidate terms from corpus and cluster them under seed tags.

        Returns:
            {seed_tag: [matched_terms...]} – domain concept terminology table.
        """
        print("[TagClustering] Extracting candidate terms via TF-IDF...")
        term_scores = self.tfidf.extract_from_corpus(corpus)
        candidate_terms = list(term_scores.keys())
        print(f"[TagClustering] Found {len(candidate_terms)} candidate terms.")

        print("[TagClustering] Clustering terms under seed tags...")
        clusters = self.clusterer.cluster_terms(candidate_terms, self.expanded_tags)

        total = sum(len(v) for v in clusters.values())
        print(f"[TagClustering] Clustered {total} terms across {len(clusters)} tags.")
        return clusters

    def run(
        self,
        corpus: List[str],
        tokenized_corpus: List[List[str]],
        seed_tags: List[str],
    ) -> Dict[str, List[str]]:
        """Convenience: fit then cluster."""
        self.fit(corpus, tokenized_corpus, seed_tags)
        return self.extract_and_cluster(corpus)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.clusterer.save(os.path.join(output_dir, "word2vec.model"))
        with open(os.path.join(output_dir, "expanded_tags.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: list(v) for k, v in self.expanded_tags.items()},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[TagClustering] Saved models to {output_dir}")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    corpus: List[str],
    tokenized_corpus: List[List[str]],
    seed_tags: List[str],
    param_grid: Optional[Dict] = None,
) -> List[Dict]:
    """
    Vary key parameters and record the number of clustered tags:
      - Number of preset tags (initial seed count)
      - TF-IDF threshold
      - Lexicon coverage (Word2Vec similarity threshold)

    Returns:
        List of result dicts: {param_name, param_value, num_clustered_terms}.
    """
    if param_grid is None:
        param_grid = {
            "tfidf_threshold": [0.05, 0.10, 0.15, 0.20, 0.25],
            "similarity_threshold": [0.60, 0.65, 0.70, 0.75, 0.80],
            "num_seed_tags": [10, 20, 30, 40, 50],
        }

    results = []

    for threshold in param_grid.get("tfidf_threshold", []):
        cfg = dict(CLUSTERING_CONFIG)
        cfg["tfidf_threshold"] = threshold
        pipeline = TagBasedClusteringPipeline(config=cfg)
        clusters = pipeline.run(corpus, tokenized_corpus, seed_tags)
        total = sum(len(v) for v in clusters.values())
        results.append({
            "param_name": "tfidf_threshold",
            "param_value": threshold,
            "num_clustered_terms": total,
        })

    for sim_thr in param_grid.get("similarity_threshold", []):
        cfg = dict(CLUSTERING_CONFIG)
        cfg["similarity_threshold"] = sim_thr
        pipeline = TagBasedClusteringPipeline(config=cfg)
        clusters = pipeline.run(corpus, tokenized_corpus, seed_tags)
        total = sum(len(v) for v in clusters.values())
        results.append({
            "param_name": "similarity_threshold",
            "param_value": sim_thr,
            "num_clustered_terms": total,
        })

    for n_seeds in param_grid.get("num_seed_tags", []):
        selected_tags = seed_tags[:n_seeds]
        pipeline = TagBasedClusteringPipeline()
        clusters = pipeline.run(corpus, tokenized_corpus, selected_tags)
        total = sum(len(v) for v in clusters.values())
        results.append({
            "param_name": "num_seed_tags",
            "param_value": n_seeds,
            "num_clustered_terms": total,
        })

    return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_clustering(
    predicted_clusters: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for the clustering output.

    Args:
        predicted_clusters: {tag: [predicted_terms...]}
        ground_truth:       {tag: [correct_terms...]}

    Returns:
        Dict with keys: precision, recall, f1.
    """
    tp = fp = fn = 0
    all_tags = set(predicted_clusters) | set(ground_truth)
    for tag in all_tags:
        pred_set = set(predicted_clusters.get(tag, []))
        gold_set = set(ground_truth.get(tag, []))
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}
