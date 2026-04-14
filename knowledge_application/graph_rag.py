"""
Graph RAG: Retrieve relevant subgraph from DDKG and convert to LLM context.

Two-stage pipeline:
  1. Entity grounding  – extract candidate entity names from free-form query
  2. Subgraph retrieval – traverse Neo4j neighbourhood up to k hops
  3. Context builder   – serialize triples as natural-language context text
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from knowledge_graph.neo4j_manager import Neo4jManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TRIPLES  = 60    # cap context length
_DEFAULT_HOPS = 2


# ---------------------------------------------------------------------------
# Entity grounding (lightweight, no model required)
# ---------------------------------------------------------------------------

class EntityGrounder:
    """
    Extract candidate entity mentions from a free-form Chinese query.

    Strategy:
      1. Chinese noun-phrase chunking via simple regex + stop-word filter
      2. Fuzzy match against Neo4j entity index (full-text search)
      3. Return top-k matched entity names
    """

    _STOP = {
        "的", "了", "在", "是", "有", "和", "与", "或", "不", "也",
        "都", "被", "把", "让", "给", "对", "从", "到", "以", "及",
        "但", "而", "所", "其", "这", "那", "哪", "什么", "怎么",
        "如何", "为什么", "请问", "告诉我", "介绍",
    }

    def __init__(self, neo4j: Neo4jManager, top_k: int = 5):
        self.neo4j = neo4j
        self.top_k = top_k

    def ground(self, query: str) -> List[str]:
        """
        Return a list of DDKG entity names most relevant to the query.
        Falls back to substring-based search if full-text index unavailable.
        """
        # 1. Extract 2-8 char candidate chunks (Chinese word-boundary heuristic)
        candidates = self._extract_candidates(query)

        matched: List[str] = []
        for cand in candidates:
            hits = self._fuzzy_search(cand)
            matched.extend(hits)
            if len(matched) >= self.top_k:
                break

        return list(dict.fromkeys(matched))[: self.top_k]  # deduplicate, keep order

    def _extract_candidates(self, text: str) -> List[str]:
        """Slide a window over the text to generate n-gram candidates."""
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        tokens = [t for t in text.split() if t and t not in self._STOP]
        candidates = []
        for n in (4, 3, 2):          # prefer longer matches
            for i in range(len(tokens) - n + 1):
                candidates.append("".join(tokens[i : i + n]))
        candidates.extend(tokens)    # single tokens as fallback
        return candidates

    def _fuzzy_search(self, keyword: str) -> List[str]:
        """Search Neo4j full-text or substring match."""
        try:
            results = self.neo4j.search_entity(keyword, limit=3)
            return [r.get("name", "") for r in results if r.get("name")]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Subgraph retrieval
# ---------------------------------------------------------------------------

class SubgraphRetriever:
    """
    Given a list of seed entities, retrieve their neighbourhood from Neo4j
    and return a flat list of (subject, predicate, object) triples.
    """

    def __init__(self, neo4j: Neo4jManager, max_hops: int = _DEFAULT_HOPS):
        self.neo4j    = neo4j
        self.max_hops = max_hops

    def retrieve(self, entities: List[str]) -> List[Tuple[str, str, str]]:
        """
        Returns deduplicated triples anchored at the given entities.
        """
        seen: set = set()
        triples: List[Tuple[str, str, str]] = []

        for entity_name in entities:
            subgraph = self.neo4j.get_knowledge_subgraph(entity_name, depth=self.max_hops)
            raw = subgraph.get("subgraph", [])
            for record in raw:
                nodes = record.get("nodes", [])
                rels  = record.get("rels", [])
                for i, rel in enumerate(rels):
                    if i + 1 < len(nodes):
                        s = nodes[i].get("name", "")
                        o = nodes[i + 1].get("name", "")
                        key = (s, rel, o)
                        if key not in seen and s and o:
                            seen.add(key)
                            triples.append(key)

            # Also get direct relations of the entity
            direct = self.neo4j.get_relations(entity_name, direction="both")
            for rel in direct:
                s = rel.get("source", entity_name)
                p = rel.get("type", "relatedTo")
                o = rel.get("target", "")
                key = (s, p, o)
                if key not in seen and o:
                    seen.add(key)
                    triples.append(key)

            if len(triples) >= _MAX_TRIPLES:
                break

        return triples[: _MAX_TRIPLES]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Serialize a list of RDF triples into a natural-language context block
    suitable for LLM prompt injection.
    """

    @staticmethod
    def build(
        triples: List[Tuple[str, str, str]],
        query: str,
        entity_attrs: Optional[List[Dict]] = None,
    ) -> str:
        """
        Convert triples + optional entity attributes to a context string.

        Returns:
            Multi-line string to be inserted into the LLM system/user prompt.
        """
        if not triples and not entity_attrs:
            return ""

        lines = ["【知识图谱检索结果】"]

        if entity_attrs:
            lines.append("实体属性：")
            for attr in entity_attrs:
                name = attr.get("name", "")
                props = {k: v for k, v in attr.items() if k != "name" and v}
                if props:
                    prop_str = "，".join(f"{k}={v}" for k, v in props.items())
                    lines.append(f"  {name}：{prop_str}")

        if triples:
            lines.append("知识三元组（主语-关系-宾语）：")
            # Map English relation names to readable Chinese
            rel_map = {
                "hasStyle": "风格为", "hasMaterial": "材质为",
                "hasFunction": "功能为", "hasColor": "颜色为",
                "hasPrice": "价格范围", "usedIn": "用于",
                "belongsTo": "属于", "composition": "由...组成",
                "mappingScheme": "对应方案", "dependency": "依赖",
                "locatedIn": "位于", "relatedTo": "相关于",
                "installsIn": "安装于", "rangeOf": "产品系列",
            }
            for s, p, o in triples:
                rel_cn = rel_map.get(p, p)
                lines.append(f"  {s} → {rel_cn} → {o}")

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph RAG pipeline
# ---------------------------------------------------------------------------

class GraphRAGRetriever:
    """
    Full Graph RAG retrieval pipeline.

    Usage:
        retriever = GraphRAGRetriever(neo4j_manager)
        context   = retriever.retrieve_context("北欧风格客厅沙发推荐")
        # Pass `context` as part of the LLM system prompt
    """

    def __init__(self, neo4j: Neo4jManager, max_hops: int = _DEFAULT_HOPS):
        self.grounder   = EntityGrounder(neo4j)
        self.retriever  = SubgraphRetriever(neo4j, max_hops=max_hops)
        self.neo4j      = neo4j

    def retrieve_context(self, query: str) -> Tuple[str, List[str]]:
        """
        Args:
            query: Free-form user question in Chinese.

        Returns:
            (context_text, grounded_entities)
            context_text  – formatted string to inject into LLM prompt
            grounded_entities – entity names found in graph
        """
        entities = self.grounder.ground(query)
        if not entities:
            return "", []

        triples = self.retriever.retrieve(entities)

        # Fetch attribute cards for the top entity
        attrs = []
        if entities:
            try:
                ent = self.neo4j.get_entity(entities[0])
                if ent:
                    attrs = [ent]
            except Exception:
                pass

        context = ContextBuilder.build(triples, query, attrs)
        return context, entities
