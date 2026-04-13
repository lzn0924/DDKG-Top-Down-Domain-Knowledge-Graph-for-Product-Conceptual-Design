"""
Entity linking and knowledge fusion.

Implements Section 2 (Technical Architecture, Step 4):
  "Entity linking addresses ambiguities through word sense disambiguation
   and coreference resolution."
  "Knowledge fusion is achieved through structure transformation and
   knowledge alignment."

Pipeline:
  1. Candidate generation  – find KB entities matching the mention
  2. Word sense disambiguation – select correct entity from candidates
  3. Coreference resolution – link pronoun/ellipsis mentions
  4. Knowledge alignment    – merge duplicate entities across sources

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 2, Step 4.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mention and KB entity types
# ---------------------------------------------------------------------------

class Mention:
    """A surface-form entity mention in text."""

    def __init__(
        self,
        text: str,
        entity_type: str,
        start: int,
        end: int,
        context: str = "",
    ):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.context = context
        self.linked_entity: Optional[str] = None   # KB entity ID after linking
        self.link_confidence: float = 0.0


class KBEntity:
    """An entity in the design domain knowledge graph."""

    def __init__(
        self,
        entity_id: str,
        canonical_name: str,
        entity_type: str,
        aliases: Optional[List[str]] = None,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.entity_id = entity_id
        self.canonical_name = canonical_name
        self.entity_type = entity_type
        self.aliases: Set[str] = set(aliases or [])
        self.aliases.add(canonical_name)
        self.description = description
        self.properties = properties or {}


# ---------------------------------------------------------------------------
# Word sense disambiguation
# ---------------------------------------------------------------------------

class WordSenseDisambiguator:
    """
    Resolves entity mention ambiguity using context similarity.

    Approach: compute TF-IDF or embedding similarity between the mention
    context window and each candidate entity description, then select the
    highest-scoring candidate.
    """

    def __init__(self, context_window: int = 50):
        self.context_window = context_window

    def disambiguate(
        self,
        mention: Mention,
        candidates: List[KBEntity],
        context: str = "",
    ) -> Optional[KBEntity]:
        """
        Select the best-matching KB entity for a mention.

        Args:
            mention:    Surface mention with surrounding context.
            candidates: KB entities that surface-match the mention.
            context:    Full sentence/document context.

        Returns:
            The most likely KB entity, or None if no good match.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        ctx = context or mention.context
        ctx_window = ctx[
            max(0, mention.start - self.context_window):
            mention.end + self.context_window
        ]

        best_entity = None
        best_score = -1.0
        for entity in candidates:
            score = self._context_overlap_score(ctx_window, entity)
            if score > best_score:
                best_score = score
                best_entity = entity

        return best_entity

    @staticmethod
    def _context_overlap_score(context: str, entity: KBEntity) -> float:
        """Jaccard overlap between context tokens and entity description tokens."""
        ctx_tokens = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", context.lower()))
        desc_tokens = set(
            re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", entity.description.lower())
        )
        desc_tokens.update(
            tok for alias in entity.aliases
            for tok in re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", alias.lower())
        )
        if not ctx_tokens or not desc_tokens:
            return 0.0
        intersection = ctx_tokens & desc_tokens
        union = ctx_tokens | desc_tokens
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Coreference resolution (rule-based for Chinese)
# ---------------------------------------------------------------------------

class CoreferenceResolver:
    """
    Simple rule-based coreference resolution for Chinese design text.

    Handles common patterns:
      - "它" / "其" / "该" (pronoun) → refers to most recent entity
      - "这款产品" / "此产品" → refers to last mentioned product
      - Nominal ellipsis ("沙发，[它]使用实木框架") 
    """

    PRONOUN_RE = re.compile(
        r"(它|其|该|此|这款?|那款?|上述|前述|所述)",
        re.UNICODE,
    )

    def resolve(
        self,
        text: str,
        mentions: List[Mention],
    ) -> List[Mention]:
        """
        Assign antecedents to pronominal/zero-pronoun mentions.

        Returns:
            Updated mention list with resolved coreferences.
        """
        resolved = []
        last_entity: Optional[str] = None
        last_type: Optional[str] = None

        for mention in sorted(mentions, key=lambda m: m.start):
            if self.PRONOUN_RE.fullmatch(mention.text):
                if last_entity is not None:
                    mention.text = last_entity
                    mention.entity_type = last_type or mention.entity_type
            else:
                last_entity = mention.text
                last_type = mention.entity_type
            resolved.append(mention)

        return resolved


# ---------------------------------------------------------------------------
# Knowledge alignment (cross-source entity merging)
# ---------------------------------------------------------------------------

class KnowledgeAligner:
    """
    Merges duplicate or equivalent entities from multiple knowledge sources.

    Strategy:
      1. Exact string match on canonical names and aliases
      2. Approximate match using edit distance for minor variations
      3. Type-constrained: only merge entities of the same type
    """

    def __init__(self, edit_dist_threshold: int = 2):
        self.edit_dist_threshold = edit_dist_threshold

    def align(self, entities_a: List[KBEntity], entities_b: List[KBEntity]) -> Dict[str, str]:
        """
        Find equivalent entity pairs between two entity sets.

        Returns:
            Dict mapping entity_id in B → entity_id in A (the merged canonical form).
        """
        alignment: Dict[str, str] = {}
        for eb in entities_b:
            best_match = self._find_match(eb, entities_a)
            if best_match:
                alignment[eb.entity_id] = best_match.entity_id
        return alignment

    def merge(
        self,
        entities_a: List[KBEntity],
        entities_b: List[KBEntity],
    ) -> List[KBEntity]:
        """Merge two entity lists, deduplicating via alignment."""
        alignment = self.align(entities_a, entities_b)
        merged = {ea.entity_id: ea for ea in entities_a}

        for eb in entities_b:
            if eb.entity_id in alignment:
                # Merge aliases and properties into the canonical entity
                canonical = merged[alignment[eb.entity_id]]
                canonical.aliases.update(eb.aliases)
                for k, v in eb.properties.items():
                    if k not in canonical.properties:
                        canonical.properties[k] = v
            else:
                merged[eb.entity_id] = eb

        return list(merged.values())

    def _find_match(
        self, entity: KBEntity, candidates: List[KBEntity]
    ) -> Optional[KBEntity]:
        for candidate in candidates:
            if candidate.entity_type != entity.entity_type:
                continue
            # Exact alias match
            if entity.aliases & candidate.aliases:
                return candidate
            # Approximate match (edit distance on canonical names)
            if self._edit_distance(
                entity.canonical_name, candidate.canonical_name
            ) <= self.edit_dist_threshold:
                return candidate
        return None

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein distance (dynamic programming)."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]


# ---------------------------------------------------------------------------
# Entity linker (top-level interface)
# ---------------------------------------------------------------------------

class EntityLinker:
    """
    Full entity linking pipeline:
      mention → candidate generation → WSD → coreference resolution → alignment
    """

    def __init__(
        self,
        knowledge_base: Optional[List[KBEntity]] = None,
        context_window: int = 50,
    ):
        self.kb: List[KBEntity] = knowledge_base or []
        self._alias_index: Dict[str, List[KBEntity]] = self._build_index()
        self.wsd = WordSenseDisambiguator(context_window=context_window)
        self.coref = CoreferenceResolver()
        self.aligner = KnowledgeAligner()

    def _build_index(self) -> Dict[str, List[KBEntity]]:
        index: Dict[str, List[KBEntity]] = {}
        for entity in self.kb:
            for alias in entity.aliases:
                index.setdefault(alias, []).append(entity)
        return index

    def add_entity(self, entity: KBEntity) -> None:
        self.kb.append(entity)
        for alias in entity.aliases:
            self._alias_index.setdefault(alias, []).append(entity)

    def link(
        self,
        mentions: List[Mention],
        context: str = "",
    ) -> List[Mention]:
        """
        Link a list of mentions to knowledge base entities.

        Args:
            mentions: Detected entity mentions from NER.
            context:  Full source text for WSD context.

        Returns:
            Mentions with .linked_entity filled in where possible.
        """
        # Coreference resolution first
        mentions = self.coref.resolve(context, mentions)

        for mention in mentions:
            candidates = self._alias_index.get(mention.text, [])
            if not candidates:
                # Partial match fallback
                candidates = [
                    ent for alias, ents in self._alias_index.items()
                    if mention.text in alias or alias in mention.text
                    for ent in ents
                    if ent.entity_type == mention.entity_type
                ]

            best = self.wsd.disambiguate(mention, candidates, context)
            if best:
                mention.linked_entity = best.entity_id
                mention.link_confidence = 1.0 if best in self._alias_index.get(
                    mention.text, []
                ) else 0.85

        return mentions

    def load_from_neo4j_export(
        self, entities_data: List[Dict[str, Any]]
    ) -> None:
        """
        Populate KB from a Neo4j graph export.

        Expected format: list of dicts with keys:
          id, name, type, aliases (list), description, properties (dict)
        """
        for data in entities_data:
            entity = KBEntity(
                entity_id=data["id"],
                canonical_name=data["name"],
                entity_type=data["type"],
                aliases=data.get("aliases", []),
                description=data.get("description", ""),
                properties=data.get("properties", {}),
            )
            self.add_entity(entity)
        self._alias_index = self._build_index()
        print(f"[EntityLinker] Loaded {len(self.kb)} entities from Neo4j export.")
