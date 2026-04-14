"""
LLM-enhanced QA for the DDKG.

Unlike the rule-based KnowledgeQAService, this service delegates
natural-language understanding to the Qwen LLM, enabling it to handle
free-form questions without fixed keyword patterns or intent label sets.

Pipeline:
  user question
    → LLM extracts entities + intent keywords  (no regex / no fixed labels)
    → graph retrieval for each entity           (Neo4j)
    → LLM synthesises a fluent answer           (context-grounded generation)
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from config import LLM_CONFIG
from knowledge_graph.neo4j_manager import Neo4jManager
from knowledge_application.llm_client import QwenClient


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYS_EXTRACT = (
    "你是一个产品概念设计领域的实体提取器。"
    "从用户的问题中提取所有实体（产品、材料、风格、功能、品牌、颜色、空间等）以及意图关键词。"
    "严格以如下 JSON 格式输出，不要有任何额外内容：\n"
    '{"entities": ["实体1", "实体2"], "intent_keywords": ["关键词1", "关键词2"]}'
)

_SYS_ANSWER = (
    "你是一个产品概念设计领域的知识图谱问答助手。"
    "以下是从知识图谱中检索到的相关事实，请根据这些事实用简洁专业的中文回答用户问题。"
    "若事实不足以支撑完整回答，请如实说明，不要编造内容。"
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LLMQAService:
    """
    LLM-augmented QA: free-form question → graph retrieval → answer.

    Compared with KnowledgeQAService:
      - No fixed intent label set or keyword patterns required
      - Handles multi-entity and compositional questions
      - Answers grounded in real graph facts (no hallucination risk from LLM alone)
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        llm_client:    Optional[QwenClient]   = None,
        config:        Optional[Dict]          = None,
    ):
        self.graph = neo4j_manager
        self.llm   = llm_client or QwenClient(config or LLM_CONFIG)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a free-form design-domain question.

        Returns::
            {
                "question": str,
                "entities": List[str],
                "facts":    List[Dict],
                "answer":   str,
            }
        """
        entities, intent_kws = self._extract_entities(question)
        facts  = self._retrieve_facts(entities, intent_kws)
        answer = self._synthesize(question, facts)
        return {
            "question": question,
            "entities": entities,
            "facts":    facts,
            "answer":   answer,
        }

    # ------------------------------------------------------------------
    # Step 1 – free-form NLU via LLM
    # ------------------------------------------------------------------

    def _extract_entities(self, question: str) -> Tuple[List[str], List[str]]:
        msgs = [
            {"role": "system", "content": _SYS_EXTRACT},
            {"role": "user",   "content": question},
        ]
        raw = self.llm.chat(msgs, temperature=0.0, max_tokens=256)
        try:
            data = json.loads(raw)
            return data.get("entities", []), data.get("intent_keywords", [])
        except json.JSONDecodeError:
            return [question], []

    # ------------------------------------------------------------------
    # Step 2 – graph retrieval
    # ------------------------------------------------------------------

    def _retrieve_facts(
        self,
        entities:    List[str],
        intent_kws:  List[str],
        per_limit:   int = 8,
    ) -> List[Dict]:
        if self.graph is None:
            raise RuntimeError(
                "Neo4j not connected. Pass a Neo4jManager instance to LLMQAService."
            )

        facts: List[Dict] = []
        seen:  set         = set()

        for ent in entities:
            node = self.graph.get_entity(ent)
            if node:
                _add(facts, seen, "entity", node, str(node))

            for rel in self.graph.get_relations(ent, direction="both")[:per_limit]:
                _add(facts, seen, "relation", rel, str(rel))

            for kw in intent_kws:
                for hit in self.graph.search_entity(f"{ent} {kw}", limit=3):
                    _add(facts, seen, "search_hit", hit, str(hit))

        return facts

    # ------------------------------------------------------------------
    # Step 3 – answer synthesis
    # ------------------------------------------------------------------

    def _synthesize(self, question: str, facts: List[Dict]) -> str:
        context = _format_context(facts)
        msgs = [
            {"role": "system", "content": _SYS_ANSWER},
            {"role": "user",
             "content": f"知识图谱事实：\n{context}\n\n用户问题：{question}"},
        ]
        return self.llm.chat(msgs, temperature=0.4, max_tokens=1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(
    facts: List[Dict],
    seen:  set,
    kind:  str,
    data:  Any,
    key:   str,
) -> None:
    if key not in seen:
        facts.append({"type": kind, "data": data})
        seen.add(key)


def _format_context(facts: List[Dict]) -> str:
    labels = {"entity": "实体", "relation": "关系", "search_hit": "搜索结果"}
    lines  = [f"[{labels.get(f['type'], f['type'])}] {f['data']}" for f in facts]
    return "\n".join(lines) if lines else "（未检索到相关事实）"
