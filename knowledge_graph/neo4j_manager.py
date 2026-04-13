"""
Neo4j graph database interface for the DDKG.

Implements Section 2 (Technical Architecture, Step 5):
  "Extracted knowledge is stored in a Neo4j graph database and
   full-text index databases."
  "The system supports both Cypher and text-based queries for
   flexible information retrieval."

Provides:
  - Entity CRUD operations
  - Relation creation
  - Cypher-based search and retrieval
  - Full-text index management
  - N-Triples bulk import

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 2, Step 5.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

from config import NEO4J_CONFIG


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------

class Neo4jManager:
    """
    Manages all Neo4j graph database operations for the DDKG.

    Usage:
        manager = Neo4jManager()
        manager.connect()
        manager.create_entity("沙发", "FurnitureProduct", {"price": 5999.0})
        manager.close()

    Or as a context manager:
        with Neo4jManager() as manager:
            manager.search_entity("沙发")
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or NEO4J_CONFIG
        self.uri = cfg["uri"]
        self.user = cfg["user"]
        self.password = cfg["password"]
        self.database = cfg.get("database", "neo4j")
        self._driver = None

    def connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self._driver.verify_connectivity()
            print(f"[Neo4j] Connected to {self.uri} (db: {self.database})")
        except ImportError:
            raise ImportError(
                "neo4j driver not installed. Run: pip install neo4j"
            )
        except Exception as e:
            raise ConnectionError(f"[Neo4j] Connection failed: {e}")

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def _session(self) -> Generator:
        if self._driver is None:
            self.connect()
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create indexes and constraints for the DDKG schema."""
        constraints = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name, e.description]",
        ]
        with self._session() as session:
            for stmt in constraints + indexes:
                try:
                    session.run(stmt)
                except Exception as e:
                    print(f"[Neo4j] Schema warning: {e}")
        print("[Neo4j] Schema initialized.")

    # ------------------------------------------------------------------
    # Entity operations
    # ------------------------------------------------------------------

    def create_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create or merge an entity node.

        Args:
            name:        Canonical entity name.
            entity_type: OWL class name (e.g., "FurnitureProduct").
            properties:  Additional node properties.

        Returns:
            Neo4j element ID of the created/merged node.
        """
        props = properties or {}
        props["name"] = name
        props["type"] = entity_type

        cypher = (
            f"MERGE (e:{entity_type} {{name: $name}}) "
            "SET e += $props "
            "RETURN elementId(e) AS id"
        )
        with self._session() as session:
            result = session.run(cypher, name=name, props=props)
            record = result.single()
            return record["id"] if record else ""

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by its canonical name."""
        cypher = "MATCH (e {name: $name}) RETURN e LIMIT 1"
        with self._session() as session:
            result = session.run(cypher, name=name)
            record = result.single()
            if record:
                return dict(record["e"])
        return None

    def delete_entity(self, name: str) -> bool:
        cypher = "MATCH (e {name: $name}) DETACH DELETE e RETURN COUNT(e) AS n"
        with self._session() as session:
            result = session.run(cypher, name=name)
            record = result.single()
            return bool(record and record["n"] > 0)

    # ------------------------------------------------------------------
    # Relation operations
    # ------------------------------------------------------------------

    def create_relation(
        self,
        head_name: str,
        relation_type: str,
        tail_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a directed relation between two entities.

        Args:
            head_name:     Canonical name of the head entity.
            relation_type: Relation type (e.g., "hasMaterial", "usedIn").
            tail_name:     Canonical name of the tail entity.
            properties:    Relation properties (e.g., confidence score).

        Returns:
            True if relation was created or already exists.
        """
        props = properties or {}
        rel_type = relation_type.upper().replace(" ", "_")
        cypher = (
            "MATCH (h {name: $head}), (t {name: $tail}) "
            f"MERGE (h)-[r:{rel_type}]->(t) "
            "SET r += $props "
            "RETURN r"
        )
        with self._session() as session:
            result = session.run(
                cypher, head=head_name, tail=tail_name, props=props
            )
            return result.single() is not None

    def get_relations(
        self,
        entity_name: str,
        relation_type: Optional[str] = None,
        direction: str = "out",    # "out" | "in" | "both"
    ) -> List[Dict[str, Any]]:
        """Retrieve relations for an entity, optionally filtered by type."""
        rel_filter = f":{relation_type.upper()}" if relation_type else ""
        if direction == "out":
            pattern = f"(e {{name: $name}})-[r{rel_filter}]->(t)"
        elif direction == "in":
            pattern = f"(t)-[r{rel_filter}]->(e {{name: $name}})"
        else:
            pattern = f"(e {{name: $name}})-[r{rel_filter}]-(t)"

        cypher = (
            f"MATCH {pattern} "
            "RETURN type(r) AS relation, t.name AS target, "
            "t.type AS target_type, properties(r) AS rel_props"
        )
        with self._session() as session:
            result = session.run(cypher, name=entity_name)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Search & retrieval (knowledge service layer)
    # ------------------------------------------------------------------

    def search_entity(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search over entity names and descriptions.

        Supports both exact and partial matches via Neo4j full-text index.
        """
        if entity_type:
            cypher = (
                f"MATCH (e:{entity_type}) "
                "WHERE e.name CONTAINS $query OR e.description CONTAINS $query "
                "RETURN e LIMIT $limit"
            )
        else:
            cypher = (
                "CALL db.index.fulltext.queryNodes('entity_fulltext', $query) "
                "YIELD node, score "
                "RETURN node AS e, score LIMIT $limit"
            )
        with self._session() as session:
            result = session.run(cypher, query=query, limit=limit)
            return [dict(record["e"]) for record in result]

    def get_knowledge_subgraph(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieve a subgraph rooted at the given entity up to `depth` hops.

        Used for exploratory analysis (Fig. 17) and knowledge recommendation.
        """
        cypher = (
            "MATCH path = (e {name: $name})-[*1..$depth]-(n) "
            "RETURN [node in nodes(path) | {name: node.name, type: node.type}] AS nodes, "
            "[rel in relationships(path) | type(rel)] AS rels"
        )
        with self._session() as session:
            result = session.run(cypher, name=entity_name, depth=depth)
            records = [dict(r) for r in result]
        return {"entity": entity_name, "subgraph": records}

    def cypher_query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute an arbitrary Cypher query and return results as dicts."""
        with self._session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Bulk import from N-Triples
    # ------------------------------------------------------------------

    def import_ntriples(
        self,
        nt_path: str,
        batch_size: int = 500,
    ) -> Tuple[int, int]:
        """
        Bulk-import triples from an N-Triples file into Neo4j.

        Parses each line as <subject> <predicate> <object> .
        Creates nodes and relationships accordingly.

        Returns:
            (nodes_created, relations_created)
        """
        RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        OWL_NI = "<http://www.w3.org/2002/07/owl#NamedIndividual>"

        import re
        triple_re = re.compile(
            r'^(<[^>]+>)\s+(<[^>]+>)\s+(<[^>]+>|"[^"]*"(?:@\w+|\^\^<[^>]+>)?)\s*\.$'
        )

        nodes = 0
        rels = 0
        batch: List[Tuple] = []

        def flush(batch_triples):
            nonlocal nodes, rels
            with self._session() as session:
                for s, p, o in batch_triples:
                    s_name = s.strip("<>").split("#")[-1].split("/")[-1]
                    p_name = p.strip("<>").split("#")[-1].split("/")[-1]
                    is_type = p == RDF_TYPE
                    is_literal = o.startswith('"')

                    if is_type and o != OWL_NI:
                        cls = o.strip("<>").split("#")[-1].split("/")[-1]
                        session.run(
                            f"MERGE (n:{cls} {{name: $name}}) "
                            "ON CREATE SET n.type = $cls",
                            name=s_name, cls=cls,
                        )
                        nodes += 1
                    elif not is_type and not is_literal:
                        o_name = o.strip("<>").split("#")[-1].split("/")[-1]
                        rel_type = p_name.upper()
                        session.run(
                            f"MERGE (h {{name: $h}}) "
                            f"MERGE (t {{name: $t}}) "
                            f"MERGE (h)-[:{rel_type}]->(t)",
                            h=s_name, t=o_name,
                        )
                        rels += 1
                    elif is_literal:
                        literal_val = o.split('"')[1]
                        session.run(
                            f"MERGE (n {{name: $name}}) "
                            f"SET n.{p_name} = $val",
                            name=s_name, val=literal_val,
                        )

        with open(nt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = triple_re.match(line)
                if m:
                    batch.append((m.group(1), m.group(2), m.group(3)))
                    if len(batch) >= batch_size:
                        flush(batch)
                        batch.clear()

        if batch:
            flush(batch)

        print(f"[Neo4j] Import complete: {nodes} nodes, {rels} relations.")
        return nodes, rels

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, int]:
        """Return basic graph statistics."""
        with self._session() as session:
            node_count = session.run("MATCH (n) RETURN COUNT(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS c").single()["c"]
            label_dist = {}
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, COUNT(n) AS c "
                "ORDER BY c DESC LIMIT 20"
            )
            for record in result:
                if record["label"]:
                    label_dist[record["label"]] = record["c"]
        return {
            "total_nodes": node_count,
            "total_relations": rel_count,
            "label_distribution": label_dist,
        }
