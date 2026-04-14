"""
GraphRAG Agent: ReAct-style LLM agent for iterative multi-hop
knowledge graph reasoning over the DDKG.

Architecture:
  The agent maintains an evidence store and an LLM reasoning loop.
  At each step the LLM selects a graph tool, the tool is executed
  against Neo4j, and the observation is fed back to the LLM.
  The loop continues until the LLM emits "Final Answer" or max_steps
  is exhausted.

  Thought → Action(tool, args) → Observation → [repeat] → Final Answer

Available tools:
  search_entity      – full-text entity search
  get_neighbors      – k-hop neighbourhood of an entity
  find_path          – shortest path between two entities
  aggregate_by_type  – enumerate entities of a given ontology class
  cypher_query       – execute an arbitrary Cypher statement
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from knowledge_graph.neo4j_manager import Neo4jManager
from knowledge_application.llm_client import QwenClient


# ---------------------------------------------------------------------------
# Evidence record
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    step:    int
    tool:    str
    args:    Dict[str, Any]
    result:  Any
    summary: str = ""

    def as_observation(self) -> str:
        body = self.summary or json.dumps(self.result, ensure_ascii=False)[:300]
        return f"[Step {self.step}][{self.tool}] {body}"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REACT_SYSTEM = """你是一个产品概念设计领域知识图谱推理智能体。
通过调用图谱工具逐步收集证据，然后回答用户提出的复杂问题。

可用工具：
  search_entity(query)               在知识图谱中搜索与 query 相关的实体
  get_neighbors(name, depth=1)       获取实体 name 的邻居节点（最多 depth 跳）
  find_path(start, end)              寻找两实体之间的最短关系路径
  aggregate_by_type(entity_type, limit=10)  按本体类型枚举实体
  cypher_query(cypher)               执行 Cypher 查询语句

每步严格按以下格式输出（不要有任何额外内容）：
Thought: <当前分析及下一步计划>
Action: <工具名>
Args: <JSON 参数，例如 {"query": "现代风格沙发"}>

当证据充足可以作答时输出：
Thought: <最终综合分析>
Final Answer: <完整答案>"""

_SYS_SUMMARIZE = "请用一句话（不超过80字）概括以下知识图谱查询结果："
_SYS_FALLBACK  = (
    "根据以下知识图谱检索证据，尽力回答用户问题。"
    "若信息确实不足请如实说明，不要编造内容。"
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GraphRAGAgent:
    """
    ReAct-based multi-hop reasoning agent over the DDKG.

    The LLM acts as the reasoning engine; Neo4j is the retrieval backend.
    Tool results are summarised and accumulated as evidence that feeds
    back into the LLM context at each reasoning step.
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        llm_client:    QwenClient,
        max_steps:     int  = 6,
        verbose:       bool = False,
    ):
        self.graph     = neo4j_manager
        self.llm       = llm_client
        self.max_steps = max_steps
        self.verbose   = verbose
        self._tools: Dict[str, Callable] = self._register_tools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the ReAct loop for *question*.

        Returns::
            {
                "question": str,
                "answer":   str,
                "steps":    int,
                "evidence": List[Evidence],
                "trace":    List[str],
            }
        """
        evidence: List[Evidence] = []
        trace:    List[str]      = []
        history:  List[Dict]     = [
            {"role": "system", "content": _REACT_SYSTEM},
            {"role": "user",   "content": f"问题：{question}"},
        ]

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(history, temperature=0.2, max_tokens=512)
            history.append({"role": "assistant", "content": reply})

            if self.verbose:
                print(f"\n── Step {step} ──\n{reply}")

            # Terminal condition
            if "Final Answer:" in reply:
                answer = reply.split("Final Answer:", 1)[-1].strip()
                trace.append(f"[Step {step}] → Final Answer")
                return self._build_result(question, answer, step, evidence, trace)

            # Parse and execute tool call
            tool_name, tool_args, thought = self._parse_action(reply)
            trace.append(f"[Step {step}] Thought: {thought[:80]}")
            trace.append(f"         Action:  {tool_name}({tool_args})")

            observation = self._execute_tool(step, tool_name, tool_args, evidence)
            trace.append(f"         Obs:     {observation[:120]}")
            history.append({"role": "user", "content": f"Observation: {observation}"})

        # Max steps reached – synthesise from accumulated evidence
        answer = self._synthesize_from_evidence(question, evidence)
        return self._build_result(question, answer, self.max_steps, evidence, trace)

    def demo_run(self, question: str) -> str:
        """Pretty-print the reasoning trace and return the final answer."""
        result = self.run(question)
        sep = "=" * 62
        print(sep)
        print(f"问题: {question}")
        print("-" * 62)
        for line in result["trace"]:
            print(line)
        print("-" * 62)
        print(f"最终答案: {result['answer']}")
        print(
            f"（共 {result['steps']} 步推理，"
            f"累积 {len(result['evidence'])} 条图谱证据）"
        )
        print(sep)
        return result["answer"]

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> Dict[str, Callable]:
        return {
            "search_entity":     self._tool_search_entity,
            "get_neighbors":     self._tool_get_neighbors,
            "find_path":         self._tool_find_path,
            "aggregate_by_type": self._tool_aggregate_by_type,
            "cypher_query":      self._tool_cypher_query,
        }

    def _tool_search_entity(self, query: str, limit: int = 5) -> List[Dict]:
        return self.graph.search_entity(query, limit=limit)

    def _tool_get_neighbors(self, name: str, depth: int = 1) -> Dict:
        return self.graph.get_knowledge_subgraph(name, depth=depth)

    def _tool_find_path(self, start: str, end: str) -> List[Dict]:
        cypher = (
            "MATCH p = shortestPath((a {name: $start})-[*1..6]-(b {name: $end})) "
            "RETURN [n IN nodes(p) | n.name] AS path, "
            "       [r IN relationships(p) | type(r)] AS rels"
        )
        return self.graph.cypher_query(cypher, {"start": start, "end": end})

    def _tool_aggregate_by_type(self, entity_type: str, limit: int = 10) -> List[Dict]:
        safe_type = re.sub(r"[^A-Za-z0-9_]", "", entity_type)
        cypher = (
            f"MATCH (n:{safe_type}) "
            "RETURN n.name AS name, labels(n) AS labels LIMIT $limit"
        )
        return self.graph.cypher_query(cypher, {"limit": limit})

    def _tool_cypher_query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        return self.graph.cypher_query(cypher, params or {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        step:      int,
        tool_name: str,
        tool_args: Dict,
        evidence:  List[Evidence],
    ) -> str:
        if tool_name not in self._tools:
            return f"[错误] 未知工具: {tool_name}"
        try:
            raw    = self._tools[tool_name](**tool_args)
            summary = self._summarize(raw)
            evidence.append(Evidence(step, tool_name, tool_args, raw, summary))
            return summary
        except Exception as exc:
            return f"[工具执行失败] {exc}"

    def _parse_action(self, text: str) -> Tuple[str, Dict, str]:
        thought   = ""
        tool_name = "search_entity"
        tool_args: Dict = {}
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Thought:"):
                thought = stripped[8:].strip()
            elif stripped.startswith("Action:"):
                tool_name = stripped[7:].strip()
            elif stripped.startswith("Args:"):
                raw = stripped[5:].strip()
                try:
                    tool_args = json.loads(raw)
                except json.JSONDecodeError:
                    tool_args = {"query": raw}
        return tool_name, tool_args, thought

    def _summarize(self, result: Any) -> str:
        if not result:
            return "（未找到相关结果）"
        text = json.dumps(result, ensure_ascii=False)[:500]
        try:
            return self.llm.chat(
                [
                    {"role": "system", "content": _SYS_SUMMARIZE},
                    {"role": "user",   "content": text},
                ],
                temperature=0.0,
                max_tokens=128,
            )
        except Exception:
            return text[:200]

    def _synthesize_from_evidence(
        self, question: str, evidence: List[Evidence]
    ) -> str:
        facts = "\n".join(e.as_observation() for e in evidence) or "（无有效检索结果）"
        return self.llm.chat(
            [
                {"role": "system", "content": _SYS_FALLBACK},
                {"role": "user",   "content": f"证据：\n{facts}\n\n问题：{question}"},
            ],
            temperature=0.4,
            max_tokens=1024,
        )

    @staticmethod
    def _build_result(
        question: str,
        answer:   str,
        steps:    int,
        evidence: List[Evidence],
        trace:    List[str],
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "answer":   answer,
            "steps":    steps,
            "evidence": evidence,
            "trace":    trace,
        }
