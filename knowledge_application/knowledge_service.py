"""
Knowledge service and management system (Section 3.2.3 and Fig. 16/17).

Provides the three core functional modes described in Section 3.2.3:
  1. Information retrieval   – search DDKG by keyword or Cypher
  2. Knowledge recommendation – suggest related entities from subgraph
  3. Exploratory analysis    – drill-down layer-by-layer navigation

Also implements the full Q&A pipeline (Section 3.2.1 / Fig. 14):
  user query → entity extraction + intent classification →
  entity linking → Cypher query generation → DDKG retrieval → answer

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 3.2.3.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from config import NER_CONFIG, SIMILARITY_CONFIG, INTENT_LABELS, MODEL_DIR
from knowledge_graph.neo4j_manager import Neo4jManager


# ---------------------------------------------------------------------------
# Cypher query templates (Fig. 14 pipeline)
# ---------------------------------------------------------------------------

CYPHER_TEMPLATES: Dict[str, str] = {
    # Intent → attribute lookup
    "design_style": (
        "MATCH (e {{name: $entity}})-[:hasStyle]->(s:StyleEntity) "
        "RETURN s.name AS result"
    ),
    "building_materials": (
        "MATCH (e {{name: $entity}})-[:hasMaterial]->(m:MaterialEntity) "
        "RETURN m.name AS result"
    ),
    "colors": (
        "MATCH (e {{name: $entity}})-[:hasColor]->(c:ColorEntity) "
        "RETURN c.name AS result"
    ),
    "furniture": (
        "MATCH (p:FurnitureProduct) WHERE p.name CONTAINS $entity "
        "RETURN p.name AS result, p.price AS price, p.description AS desc"
    ),
    "appliance_configuration": (
        "MATCH (p:ApplianceProduct) WHERE p.name CONTAINS $entity "
        "RETURN p.name AS result, p.description AS desc"
    ),
    "spatial_structure": (
        "MATCH (s:SpatialStructure) WHERE s.name CONTAINS $entity "
        "RETURN s.name AS result, s.description AS desc"
    ),
    "product_models": (
        "MATCH (p:Product) WHERE p.name CONTAINS $entity "
        "RETURN p.name AS result, p.modelNumber AS model"
    ),
    "design_cases": (
        "MATCH (s:SchemeConceptionKnowledge) WHERE s.name CONTAINS $entity "
        "RETURN s.name AS result, s.description AS desc"
    ),
    # Generic fallback
    "default": (
        "MATCH (n) WHERE n.name CONTAINS $entity "
        "RETURN n.name AS result, n.type AS type LIMIT 10"
    ),
}


# ---------------------------------------------------------------------------
# Knowledge Q&A service
# ---------------------------------------------------------------------------

class KnowledgeQAService:
    """
    Q&A pipeline based on the top-down hierarchical DDKG (Fig. 14).

    Step 1: Extract design knowledge subjects (entities) and query intent
            from user question via the joint QA model.
    Step 2: Link extracted entities to DDKG nodes.
    Step 3: Convert query intent + entities into a Cypher query.
    Step 4: Execute Cypher against Neo4j and return structured results.
    """

    def __init__(
        self,
        qa_model=None,
        neo4j_manager: Optional[Neo4jManager] = None,
        entity_linker=None,
        device: Optional[str] = None,
    ):
        self.qa_model = qa_model
        self.neo4j = neo4j_manager
        self.linker = entity_linker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(NER_CONFIG["bert_model"])
        return self._tokenizer

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a natural language design knowledge query.

        Args:
            question: User's Chinese query string.

        Returns:
            Dict with keys:
              entities:  list of (entity_text, entity_type)
              intent:    predicted query intent string
              cypher:    generated Cypher query
              results:   list of retrieved records from Neo4j
              answer:    human-readable summary
        """
        # Step 1: Joint extraction (entities + intent)
        entities, intent = self._extract_query_info(question)

        # Step 2: Entity linking
        entity_names = [e[0] for e in entities]
        linked = self._link_entities(entity_names)

        # Step 3: Cypher generation
        cypher, params = self._build_cypher(intent, linked)

        # Step 4: Execute against DDKG
        results = self._execute_query(cypher, params)

        # Step 5: Format answer
        answer = self._format_answer(intent, entities, results)

        return {
            "question": question,
            "entities": entities,
            "intent": intent,
            "linked_entities": linked,
            "cypher": cypher,
            "results": results,
            "answer": answer,
        }

    def _extract_query_info(
        self, question: str
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Use the joint QA model to extract entities and intent."""
        if self.qa_model is None:
            raise RuntimeError(
                "No QA model loaded. Pass a trained JointQAModel instance "
                "to KnowledgeQAService(qa_model=...) before calling answer()."
            )
        from knowledge_application.qa_model import QATrainer
        tokenizer = self._get_tokenizer()
        trainer = QATrainer(self.qa_model, device=self.device)
        output = trainer.predict(question, tokenizer)
        return output["entities"], output["intent"]

    def _link_entities(self, entity_names: List[str]) -> List[str]:
        """Link entity surface forms to canonical DDKG node names."""
        if self.linker is None or not entity_names:
            return entity_names
        from knowledge_extraction.entity_linking import Mention
        mentions = [
            Mention(text=name, entity_type="ENTITY", start=0, end=len(name))
            for name in entity_names
        ]
        linked = self.linker.link(mentions)
        return [
            m.linked_entity if m.linked_entity else m.text
            for m in linked
        ]

    def _build_cypher(
        self,
        intent: str,
        entities: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Select and parameterize a Cypher template for the given intent."""
        template = CYPHER_TEMPLATES.get(intent, CYPHER_TEMPLATES["default"])
        entity_str = entities[0] if entities else ""
        params = {"entity": entity_str}
        return template, params

    def _execute_query(
        self,
        cypher: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query against Neo4j."""
        if self.neo4j is None:
            return [{"info": "Neo4j not connected. Configure NEO4J_CONFIG in config.py."}]
        try:
            return self.neo4j.cypher_query(cypher, params)
        except Exception as e:
            return [{"error": str(e)}]

    @staticmethod
    def _format_answer(
        intent: str,
        entities: List[Tuple[str, str]],
        results: List[Dict[str, Any]],
    ) -> str:
        """Format retrieved results into a human-readable answer."""
        if not results:
            return "未找到相关知识。请尝试更换关键词。"
        entity_str = "、".join(e[0] for e in entities) if entities else "该产品"
        result_str = "；".join(
            str(v) for rec in results for v in rec.values() if v
        )
        intent_str = intent.replace('_', ' ')
        return f'关于\u201c{entity_str}\u201d的{intent_str}信息：{result_str}'


# ---------------------------------------------------------------------------
# Knowledge retrieval service (Fig. 16)
# ---------------------------------------------------------------------------

class KnowledgeRetrievalService:
    """
    Main functional interface of the knowledge service system (Fig. 16).

    Features:
      - Entity search with attribute display
      - Associated entity analysis
      - Domain knowledge extension via subgraph traversal
    """

    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        self.neo4j = neo4j_manager

    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search the DDKG by keyword.

        Returns entity entries matching the search criteria,
        along with attribute information of each entity.
        """
        if self.neo4j is None:
            raise RuntimeError(
                "Neo4j not connected. Pass a connected Neo4jManager instance "
                "to KnowledgeRetrievalService(neo4j_manager=...)."
            )
        return self.neo4j.search_entity(query, entity_type=entity_type, limit=limit)

    def get_entity_with_associations(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieve an entity and its associated knowledge entities.

        "By analyzing the associated entities of the knowledge entity, the system
        can provide users with domain knowledge extensions." (Section 3.2.3)
        """
        if self.neo4j is None:
            return {"entity": entity_name, "associations": []}

        entity = self.neo4j.get_entity(entity_name)
        relations = self.neo4j.get_relations(entity_name, direction="both")
        subgraph = self.neo4j.get_knowledge_subgraph(entity_name, depth=depth)

        return {
            "entity": entity,
            "direct_relations": relations,
            "subgraph": subgraph,
        }

    def recommend(
        self,
        entity_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend related design knowledge based on graph neighborhood.

        Surfaces entities connected within 2 hops that are not already known,
        ranked by connectivity (number of shared neighbors).
        """
        if self.neo4j is None:
            raise RuntimeError(
                "Neo4j not connected. Pass a connected Neo4jManager instance."
            )
        cypher = (
            "MATCH (e {name: $name})-[*1..2]-(related) "
            "WHERE related.name <> $name "
            "WITH related, COUNT(*) AS conn_score "
            "ORDER BY conn_score DESC "
            "RETURN related.name AS name, related.type AS type, "
            "conn_score LIMIT $top_k"
        )
        return self.neo4j.cypher_query(cypher, {"name": entity_name, "top_k": top_k})


# ---------------------------------------------------------------------------
# Semantic similarity service (uses contrastive model)
# ---------------------------------------------------------------------------

class SemanticSimilarityService:
    """
    Semantic similarity calculation for knowledge retrieval ranking.

    "BERT-WWM is employed to calculate semantic similarity,
    enhancing the relevance of search results." (Section 2, Step 6)

    Uses the trained contrastive model (Appendix C) to encode queries
    and candidate documents, then ranks by cosine similarity.
    """

    def __init__(
        self,
        similarity_model=None,
        tokenizer=None,
        device: Optional[str] = None,
    ):
        self.model = similarity_model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.tokenizer is None and self.model is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                SIMILARITY_CONFIG["bert_model"]
            )

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        if self.model is None:
            return 0.0

        self.model.eval()
        enc_a = self.tokenizer(
            text_a,
            max_length=SIMILARITY_CONFIG["max_seq_len"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            text_b,
            max_length=SIMILARITY_CONFIG["max_seq_len"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids_a=enc_a["input_ids"].to(self.device),
                attention_mask_a=enc_a["attention_mask"].to(self.device),
                input_ids_b=enc_b["input_ids"].to(self.device),
                attention_mask_b=enc_b["attention_mask"].to(self.device),
            )
        return float(outputs["similarity"].item())

    def rank_candidates(
        self,
        query: str,
        candidates: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate texts by semantic similarity to the query.

        Returns:
            List of (candidate_text, similarity_score) sorted descending.
        """
        scored = [(cand, self.compute_similarity(query, cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
