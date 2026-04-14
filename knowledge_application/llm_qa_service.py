"""
LLM-augmented Q&A service with Graph RAG.

Combines:
  1. GraphRAGRetriever  – pulls relevant subgraph context from Neo4j
  2. QwenClient         – sends context + query to AutoDL-deployed Qwen LLM
  3. Conversation memory – multi-turn dialogue with rolling history window

Unlike the keyword-intent pipeline in knowledge_service.py, this service
handles arbitrary free-form questions and is not limited to predefined intents.
"""

from typing import Any, Dict, List, Optional, Tuple

from knowledge_application.llm_client  import QwenClient, LLM_CONFIG, _SYSTEM_PROMPT
from knowledge_application.graph_rag   import GraphRAGRetriever
from knowledge_graph.neo4j_manager     import Neo4jManager

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_WITH_CONTEXT = """{base_system}

以下是从产品设计领域知识图谱中检索到的背景知识，请优先基于此回答：

{context}
"""

_SYSTEM_NO_CONTEXT = """{base_system}

当前知识图谱中未检索到与问题直接相关的条目，请基于你的通用知识回答，
并提示用户该信息未经知识图谱验证。
"""


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class LLMQAService:
    """
    Multi-turn Q&A service backed by Graph RAG + Qwen LLM.

    Each session maintains a rolling conversation history so follow-up
    questions can reference previous context.

    Example:
        service = LLMQAService(neo4j_manager=mgr)
        answer  = service.ask("北欧风格客厅适合用什么材质的沙发？")
        answer2 = service.ask("价格一般在什么范围？")  # follow-up
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        llm_config:    Optional[Dict]          = None,
        max_history:   int                     = 6,
        max_hops:      int                     = 2,
    ):
        self.llm     = QwenClient(llm_config or LLM_CONFIG)
        self.rag     = GraphRAGRetriever(neo4j_manager, max_hops=max_hops) \
                       if neo4j_manager else None
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history      # rolling window (user+assistant pairs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, query: str, use_graph: bool = True) -> Dict[str, Any]:
        """
        Answer a free-form question.

        Args:
            query:     User question (Chinese or English).
            use_graph: Whether to perform Graph RAG retrieval.

        Returns:
            {
              "answer":   str,
              "entities": List[str],   # grounded entities
              "context_used": bool,    # whether graph context was injected
            }
        """
        context_text = ""
        entities: List[str] = []

        # 1. Graph RAG retrieval
        if use_graph and self.rag:
            context_text, entities = self.rag.retrieve_context(query)

        # 2. Build system prompt
        if context_text:
            system = _SYSTEM_WITH_CONTEXT.format(
                base_system=_SYSTEM_PROMPT,
                context=context_text,
            )
        else:
            system = _SYSTEM_NO_CONTEXT.format(base_system=_SYSTEM_PROMPT)

        # 3. Compose messages: system + rolling history + new query
        messages = [{"role": "system", "content": system}]
        messages.extend(self._get_history_window())
        messages.append({"role": "user", "content": query})

        # 4. LLM call
        answer = self.llm.chat(messages)

        # 5. Update history
        self._push_history(query, answer)

        return {
            "answer":       answer,
            "entities":     entities,
            "context_used": bool(context_text),
        }

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _push_history(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user",      "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        # Keep rolling window: 2 messages per turn × max_history turns
        cap = self.max_history * 2
        if len(self.history) > cap:
            self.history = self.history[-cap:]

    def _get_history_window(self) -> List[Dict[str, str]]:
        return list(self.history)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def run_llm_qa_demo(neo4j_manager: Optional[Neo4jManager] = None) -> None:
    """Interactive command-line demo for the LLM Q&A service."""
    service = LLMQAService(neo4j_manager=neo4j_manager)

    if not service.llm.health_check():
        print("[LLM QA] Warning: LLM endpoint unreachable. Check AUTODL_LLM_URL.")
        print("         Set env var or update LLM_CONFIG in llm_client.py.\n")

    print("=" * 60)
    print("DDKG × Qwen  Graph-RAG 问答系统")
    print("输入 'quit' 退出  |  输入 'reset' 清空对话记录")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            service.reset()
            print("[系统] 对话记录已清空。")
            continue

        result = service.ask(user_input)

        print(f"\n助手: {result['answer']}")
        if result["entities"]:
            print(f"[图谱命中实体: {', '.join(result['entities'])}]")
        if not result["context_used"]:
            print("[提示: 本回答未使用知识图谱，仅供参考]")
