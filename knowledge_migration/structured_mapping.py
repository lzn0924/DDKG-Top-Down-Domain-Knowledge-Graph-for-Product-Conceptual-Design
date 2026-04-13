"""
Structured data mapping: Relational DB → Knowledge Graph.

R2RML-style mapping rules:
  Tables   → OWL classes
  Columns  → data properties (attributes)
  Rows     → named individuals (entities)
  FK links → object properties (relationships)

Generates N-Triples format output for Neo4j bulk import.
"""

import csv
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from config import DATA_DIR, ONTOLOGY_CONFIG


# ---------------------------------------------------------------------------
# Mapping rule definitions (R2RML-inspired)
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """Maps a DB column to an OWL property (data or object)."""
    column_name: str
    property_uri: str
    property_type: str = "data"    # "data" | "object"
    target_class: Optional[str] = None  # For object properties (FK targets)
    language: Optional[str] = None      # e.g., "zh" for Chinese literals


@dataclass
class TableMapping:
    """Maps a DB table to an OWL class with its column mappings."""
    table_name: str
    class_uri: str
    id_column: str                  # Primary key column
    column_mappings: List[ColumnMapping] = field(default_factory=list)
    subject_template: str = "{id_column}"  # IRI template using column values


@dataclass
class ForeignKeyMapping:
    """Maps a FK relationship to an OWL object property."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    property_uri: str


# ---------------------------------------------------------------------------
# Default mapping for home design DB schema
# ---------------------------------------------------------------------------

NS = ONTOLOGY_CONFIG["namespace"]

DEFAULT_TABLE_MAPPINGS: List[TableMapping] = [
    TableMapping(
        table_name="products",
        class_uri=f"{NS}Product",
        id_column="product_id",
        column_mappings=[
            ColumnMapping("product_name", f"{NS}hasName", "data", language="zh"),
            ColumnMapping("description", f"{NS}hasDescription", "data", language="zh"),
            ColumnMapping("price", f"{NS}hasPrice", "data"),
            ColumnMapping("model_number", f"{NS}hasModelNumber", "data"),
            ColumnMapping("manufacturer", f"{NS}hasManufacturer", "data"),
            ColumnMapping("material_id", f"{NS}hasMaterial", "object", target_class=f"{NS}MaterialEntity"),
            ColumnMapping("style_id", f"{NS}hasStyle", "object", target_class=f"{NS}StyleEntity"),
        ],
        subject_template=f"{NS}product_{{product_id}}",
    ),
    TableMapping(
        table_name="materials",
        class_uri=f"{NS}MaterialEntity",
        id_column="material_id",
        column_mappings=[
            ColumnMapping("material_name", f"{NS}hasName", "data", language="zh"),
            ColumnMapping("description", f"{NS}hasDescription", "data", language="zh"),
        ],
        subject_template=f"{NS}material_{{material_id}}",
    ),
    TableMapping(
        table_name="styles",
        class_uri=f"{NS}StyleEntity",
        id_column="style_id",
        column_mappings=[
            ColumnMapping("style_name", f"{NS}hasName", "data", language="zh"),
        ],
        subject_template=f"{NS}style_{{style_id}}",
    ),
    TableMapping(
        table_name="customers",
        class_uri=f"{NS}Customer",
        id_column="customer_id",
        column_mappings=[
            ColumnMapping("customer_name", f"{NS}hasName", "data", language="zh"),
            ColumnMapping("province", f"{NS}locatedInProvince", "data"),
            ColumnMapping("city", f"{NS}locatedInCity", "data"),
        ],
        subject_template=f"{NS}customer_{{customer_id}}",
    ),
    TableMapping(
        table_name="design_schemes",
        class_uri=f"{NS}SchemeConceptionKnowledge",
        id_column="scheme_id",
        column_mappings=[
            ColumnMapping("scheme_name", f"{NS}hasName", "data", language="zh"),
            ColumnMapping("description", f"{NS}hasDescription", "data", language="zh"),
            ColumnMapping("product_id", f"{NS}composition", "object", target_class=f"{NS}Product"),
        ],
        subject_template=f"{NS}scheme_{{scheme_id}}",
    ),
]

DEFAULT_FK_MAPPINGS: List[ForeignKeyMapping] = [
    ForeignKeyMapping("products", "material_id", "materials", "material_id", f"{NS}hasMaterial"),
    ForeignKeyMapping("products", "style_id", "styles", "style_id", f"{NS}hasStyle"),
    ForeignKeyMapping("design_schemes", "product_id", "products", "product_id", f"{NS}composition"),
]


# ---------------------------------------------------------------------------
# Triple generator
# ---------------------------------------------------------------------------

class StructuredKnowledgeMigrator:
    """
    Converts relational database records to RDF triples.

    Output format: N-Triples (.nt) – compatible with Neo4j and any RDF store.
    """

    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    OWL_NAMED_INDIVIDUAL = "http://www.w3.org/2002/07/owl#NamedIndividual"
    XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"
    XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"
    XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"

    def __init__(
        self,
        table_mappings: Optional[List[TableMapping]] = None,
        fk_mappings: Optional[List[ForeignKeyMapping]] = None,
    ):
        self.table_mappings = {m.table_name: m for m in (table_mappings or DEFAULT_TABLE_MAPPINGS)}
        self.fk_mappings = fk_mappings or DEFAULT_FK_MAPPINGS

    # ------------------------------------------------------------------
    # Triple generation
    # ------------------------------------------------------------------

    def _format_subject(self, mapping: TableMapping, row: Dict[str, Any]) -> str:
        template = mapping.subject_template
        for col_name, col_val in row.items():
            template = template.replace(f"{{{col_name}}}", str(col_val or ""))
        return f"<{template}>"

    @staticmethod
    def _format_literal(value: Any, language: Optional[str] = None) -> str:
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        if language:
            return f'"{text}"@{language}'
        try:
            float(value)
            return f'"{text}"^^<http://www.w3.org/2001/XMLSchema#decimal>'
        except (ValueError, TypeError):
            return f'"{text}"^^<http://www.w3.org/2001/XMLSchema#string>'

    def generate_triples_from_rows(
        self,
        table_name: str,
        rows: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Yield N-Triple strings for each row in a table."""
        mapping = self.table_mappings.get(table_name)
        if mapping is None:
            return

        for row in rows:
            subject = self._format_subject(mapping, row)
            # rdf:type
            yield f"{subject} <{self.RDF_TYPE}> <{mapping.class_uri}> ."
            yield f"{subject} <{self.RDF_TYPE}> <{self.OWL_NAMED_INDIVIDUAL}> ."

            for col_map in mapping.column_mappings:
                value = row.get(col_map.column_name)
                if value is None:
                    continue

                if col_map.property_type == "data":
                    obj = self._format_literal(value, col_map.language)
                    yield f"{subject} <{col_map.property_uri}> {obj} ."
                elif col_map.property_type == "object":
                    # Foreign key → object property
                    # The target IRI is built using the referenced table mapping
                    target_table = next(
                        (m.table_name for m in self.table_mappings.values()
                         if m.class_uri == col_map.target_class),
                        None,
                    )
                    if target_table:
                        target_mapping = self.table_mappings[target_table]
                        target_row = {target_mapping.id_column: value}
                        target_subject = self._format_subject(target_mapping, target_row)
                        yield (
                            f"{subject} <{col_map.property_uri}> "
                            f"{target_subject} ."
                        )

    # ------------------------------------------------------------------
    # SQLite source
    # ------------------------------------------------------------------

    def migrate_from_sqlite(
        self,
        db_path: str,
        output_path: str,
    ) -> int:
        """
        Read all mapped tables from a SQLite database and write N-Triples.

        Args:
            db_path:     Path to SQLite .db file.
            output_path: Output .nt file path.

        Returns:
            Total number of triples generated.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        total = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            for table_name in self.table_mappings:
                try:
                    cursor = conn.execute(f"SELECT * FROM {table_name}")
                    rows = [dict(row) for row in cursor.fetchall()]
                    for triple in self.generate_triples_from_rows(table_name, rows):
                        out.write(triple + "\n")
                        total += 1
                except sqlite3.OperationalError as e:
                    print(f"[StructuredMapping] Skipped table '{table_name}': {e}")

        conn.close()
        print(f"[StructuredMapping] Generated {total} triples → {output_path}")
        return total

    # ------------------------------------------------------------------
    # CSV / JSONL source
    # ------------------------------------------------------------------

    def migrate_from_csv(
        self,
        csv_path: str,
        table_name: str,
        output_path: str,
    ) -> int:
        """Migrate a CSV file as if it were a relational table."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as out:
            for triple in self.generate_triples_from_rows(table_name, rows):
                out.write(triple + "\n")
                total += 1
        print(f"[StructuredMapping] {table_name}: {total} triples from CSV → {output_path}")
        return total

    def migrate_from_jsonl(
        self,
        jsonl_path: str,
        table_name: str,
        output_path: str,
    ) -> int:
        """Migrate a JSONL file (one record per line) as a relational table."""
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        total = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as out:
            for triple in self.generate_triples_from_rows(table_name, rows):
                out.write(triple + "\n")
                total += 1
        print(f"[StructuredMapping] {table_name}: {total} triples from JSONL → {output_path}")
        return total

    # ------------------------------------------------------------------
    # Manual mapping adjustment
    # ------------------------------------------------------------------

    def add_custom_mapping(
        self,
        subject_uri: str,
        property_uri: str,
        object_value: str,
        is_literal: bool = True,
        language: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Manually add a custom triple to adapt to evolving business ontologies.

        "Mapping can also be customized and supplemented manually according to
        new business requirements, such as adding new entity types or relationships
        to better adapt to the evolution of business ontologies."
        """
        subject = f"<{subject_uri}>"
        predicate = f"<{property_uri}>"
        if is_literal:
            obj = self._format_literal(object_value, language)
        else:
            obj = f"<{object_value}>"
        triple = f"{subject} {predicate} {obj} ."

        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(triple + "\n")
        return triple
