"""
OWL ontology construction for the product conceptual design domain.

Implements the bi-level architecture (Section 2.1.2):
  - Data layer:   fundamental concepts and taxonomy (semantic network)
  - Schema layer: architectural knowledge with OWL/RDF constraints and rules

The ontology is Protégé-compatible (exported as RDF/XML .owl file).
Top-level taxonomy derived from Fig. 4 and Fig. 7 (home design ontology).

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 2.1.2.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from owlready2 import (
        get_ontology,
        Thing,
        ObjectProperty,
        DataProperty,
        FunctionalProperty,
        InverseFunctionalProperty,
        TransitiveProperty,
        SymmetricProperty,
        AllDisjoint,
        sync_reasoner_pellet,
    )
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False
    print("[OntologyBuilder] owlready2 not installed. Using lightweight RDF/XML writer.")

from config import ONTOLOGY_CONFIG, ONTOLOGY_CLASSES, RELATION_TYPES, DATA_DIR


# ---------------------------------------------------------------------------
# RDF/XML fallback writer (no owlready2 dependency)
# ---------------------------------------------------------------------------

class RDFXMLWriter:
    """
    Minimal RDF/XML serializer for OWL ontologies.
    Used when owlready2 is not available.
    """

    OWL_NS = "http://www.w3.org/2002/07/owl#"
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
    XSD_NS = "http://www.w3.org/2001/XMLSchema#"

    def __init__(self, namespace: str):
        self.ns = namespace
        self._classes: List[Dict] = []
        self._object_properties: List[Dict] = []
        self._data_properties: List[Dict] = []
        self._individuals: List[Dict] = []

    def add_class(
        self,
        name: str,
        parent: Optional[str] = None,
        comment: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        self._classes.append({
            "name": name, "parent": parent,
            "comment": comment, "label": label,
        })

    def add_object_property(
        self,
        name: str,
        domain: Optional[str] = None,
        range_: Optional[str] = None,
        inverse_of: Optional[str] = None,
        is_transitive: bool = False,
        is_symmetric: bool = False,
        comment: Optional[str] = None,
    ) -> None:
        self._object_properties.append({
            "name": name, "domain": domain, "range": range_,
            "inverse_of": inverse_of, "is_transitive": is_transitive,
            "is_symmetric": is_symmetric, "comment": comment,
        })

    def add_data_property(
        self,
        name: str,
        domain: Optional[str] = None,
        range_xsd: str = "string",
        comment: Optional[str] = None,
    ) -> None:
        self._data_properties.append({
            "name": name, "domain": domain,
            "range_xsd": range_xsd, "comment": comment,
        })

    def add_individual(
        self,
        name: str,
        class_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._individuals.append({
            "name": name, "class": class_name,
            "properties": properties or {},
        })

    def serialize(self, output_path: str) -> None:
        lines = [
            '<?xml version="1.0"?>',
            f'<rdf:RDF xmlns="{self.ns}"',
            f'         xmlns:owl="{self.OWL_NS}"',
            f'         xmlns:rdf="{self.RDF_NS}"',
            f'         xmlns:rdfs="{self.RDFS_NS}"',
            f'         xmlns:xsd="{self.XSD_NS}"',
            f'         xml:base="{self.ns}">',
            '',
            f'    <owl:Ontology rdf:about="{self.ns}"/>',
            '',
        ]

        # Classes
        for cls in self._classes:
            uri = f"{self.ns}{cls['name']}"
            lines.append(f'    <owl:Class rdf:about="{uri}">')
            if cls["parent"]:
                parent_uri = f"{self.ns}{cls['parent']}"
                lines.append(f'        <rdfs:subClassOf rdf:resource="{parent_uri}"/>')
            if cls.get("label"):
                lines.append(f'        <rdfs:label xml:lang="en">{cls["label"]}</rdfs:label>')
            if cls.get("comment"):
                lines.append(f'        <rdfs:comment>{cls["comment"]}</rdfs:comment>')
            lines.append('    </owl:Class>')
            lines.append('')

        # Object properties
        for prop in self._object_properties:
            uri = f"{self.ns}{prop['name']}"
            prop_types = ["owl:ObjectProperty"]
            if prop["is_transitive"]:
                prop_types.append("owl:TransitiveProperty")
            if prop["is_symmetric"]:
                prop_types.append("owl:SymmetricProperty")
            for prop_type in prop_types:
                lines.append(f'    <{prop_type} rdf:about="{uri}">')
                if prop.get("domain"):
                    d_uri = f"{self.ns}{prop['domain']}"
                    lines.append(f'        <rdfs:domain rdf:resource="{d_uri}"/>')
                if prop.get("range"):
                    r_uri = f"{self.ns}{prop['range']}"
                    lines.append(f'        <rdfs:range rdf:resource="{r_uri}"/>')
                if prop.get("inverse_of"):
                    inv_uri = f"{self.ns}{prop['inverse_of']}"
                    lines.append(f'        <owl:inverseOf rdf:resource="{inv_uri}"/>')
                if prop.get("comment"):
                    lines.append(f'        <rdfs:comment>{prop["comment"]}</rdfs:comment>')
                lines.append(f'    </{prop_type}>')
                lines.append('')

        # Data properties
        for prop in self._data_properties:
            uri = f"{self.ns}{prop['name']}"
            lines.append(f'    <owl:DatatypeProperty rdf:about="{uri}">')
            if prop.get("domain"):
                d_uri = f"{self.ns}{prop['domain']}"
                lines.append(f'        <rdfs:domain rdf:resource="{d_uri}"/>')
            xsd_uri = f"{self.XSD_NS}{prop['range_xsd']}"
            lines.append(f'        <rdfs:range rdf:resource="{xsd_uri}"/>')
            if prop.get("comment"):
                lines.append(f'        <rdfs:comment>{prop["comment"]}</rdfs:comment>')
            lines.append('    </owl:DatatypeProperty>')
            lines.append('')

        # Individuals
        for ind in self._individuals:
            class_uri = f"{self.ns}{ind['class']}"
            ind_uri = f"{self.ns}{ind['name']}"
            lines.append(f'    <owl:NamedIndividual rdf:about="{ind_uri}">')
            lines.append(f'        <rdf:type rdf:resource="{class_uri}"/>')
            for prop_name, value in ind["properties"].items():
                prop_uri = f"{self.ns}{prop_name}"
                lines.append(f'        <{prop_name}>{value}</{prop_name}>')
            lines.append('    </owl:NamedIndividual>')
            lines.append('')

        lines.append('</rdf:RDF>')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Domain ontology definition
# ---------------------------------------------------------------------------

class DDKGOntologyBuilder:
    """
    Constructs the Product Conceptual Design Domain Knowledge Graph ontology.

    Taxonomy (Fig. 4 / Fig. 7 / Fig. 13):
      Thing
      └── ProductConceptualDesignKnowledge
          ├── UserPreferenceKnowledge
          │   ├── PricePreference
          │   ├── FunctionPreference
          │   ├── MaterialPreference
          │   └── StylePreference
          ├── DesignObjectKnowledge
          │   ├── Product
          │   │   ├── FurnitureProduct
          │   │   └── ApplianceProduct
          │   ├── MaterialEntity
          │   ├── StyleEntity
          │   ├── SpatialStructure
          │   └── FunctionEntity
          ├── DesignMethodKnowledge
          ├── SchemeConceptionKnowledge
          ├── DesignEvaluationKnowledge
          └── ProductionProcessKnowledge

    Relation types (Table 6 in paper):
      mappingScheme (reversible), comparison (reversible),
      dependency, composition, attribute, usedIn, hasStyle, ...
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ONTOLOGY_CONFIG
        self.ns = self.config["namespace"]
        self.output_path = self.config["output_path"]
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._writer = RDFXMLWriter(self.ns)

    def _define_classes(self) -> None:
        """Define the full class taxonomy (top-down hierarchy)."""
        # Root design knowledge class
        self._writer.add_class(
            "ProductConceptualDesignKnowledge",
            parent=None,
            label="Product Conceptual Design Knowledge",
            comment="Root class for all product conceptual design domain knowledge.",
        )

        # ── User preference knowledge
        self._writer.add_class(
            "UserPreferenceKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="User Preference Knowledge",
        )
        for sub in ["PricePreference", "FunctionPreference",
                    "MaterialPreference", "StylePreference"]:
            self._writer.add_class(sub, parent="UserPreferenceKnowledge")

        # ── Design object knowledge
        self._writer.add_class(
            "DesignObjectKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="Design Object Knowledge",
        )
        self._writer.add_class("Product", parent="DesignObjectKnowledge")
        self._writer.add_class("FurnitureProduct", parent="Product",
                               comment="Sofa, bed, wardrobe, cabinet, desk, etc.")
        self._writer.add_class("ApplianceProduct", parent="Product",
                               comment="Refrigerator, washing machine, air conditioner, etc.")
        self._writer.add_class("MaterialEntity", parent="DesignObjectKnowledge")
        self._writer.add_class("StyleEntity", parent="DesignObjectKnowledge")
        self._writer.add_class("SpatialStructure", parent="DesignObjectKnowledge")
        self._writer.add_class("FunctionEntity", parent="DesignObjectKnowledge")
        self._writer.add_class("ColorEntity", parent="DesignObjectKnowledge")
        self._writer.add_class("BrandEntity", parent="DesignObjectKnowledge")

        # ── Design method knowledge
        self._writer.add_class(
            "DesignMethodKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="Design Method Knowledge",
        )

        # ── Scheme conception knowledge (Fig. 13 relations)
        self._writer.add_class(
            "SchemeConceptionKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="Scheme Conception Knowledge",
            comment="Scheme conception knowledge instances (Scheme A, Scheme B, ...).",
        )
        self._writer.add_class("DesignScheme", parent="SchemeConceptionKnowledge")

        # ── Design evaluation knowledge
        self._writer.add_class(
            "DesignEvaluationKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="Design Evaluation Knowledge",
        )

        # ── Production process knowledge
        self._writer.add_class(
            "ProductionProcessKnowledge",
            parent="ProductConceptualDesignKnowledge",
            label="Production Process Knowledge",
        )

        # Geo-entities (for customer location mapping, Table 7)
        self._writer.add_class("Customer", parent=None, label="Customer")
        self._writer.add_class("Province", parent=None, label="Province")
        self._writer.add_class("City", parent=None, label="City")
        self._writer.add_class("FunctionPart", parent=None, label="Function Part")

    def _define_object_properties(self) -> None:
        """Define object properties with domain/range constraints (schema layer)."""

        # ── Scheme conception relations (Fig. 13 / Table 6)
        self._writer.add_object_property(
            "mappingScheme",
            domain="SchemeConceptionKnowledge",
            range_="SchemeConceptionKnowledge",
            is_symmetric=True,
            comment="Scheme conception knowledge is related to each other (reversible).",
        )
        self._writer.add_object_property(
            "comparison",
            domain="SchemeConceptionKnowledge",
            range_="SchemeConceptionKnowledge",
            is_symmetric=True,
            comment="Same-type scheme comparison (reversible).",
        )
        self._writer.add_object_property(
            "dependency",
            domain="SchemeConceptionKnowledge",
            range_="DesignEvaluationKnowledge",
            comment="Correlation between scheme and design evaluation.",
        )
        self._writer.add_object_property(
            "composition",
            domain="SchemeConceptionKnowledge",
            range_="DesignObjectKnowledge",
            comment="Scheme conception knowledge includes design object knowledge.",
        )
        self._writer.add_object_property(
            "attribute",
            domain="SchemeConceptionKnowledge",
            range_="DesignMethodKnowledge",
            comment="Relationship between scheme and design method.",
        )

        # ── Product relations
        self._writer.add_object_property(
            "usedIn",
            domain="MaterialEntity",
            range_="SpatialStructure",
            comment="Material used in a particular space.",
        )
        self._writer.add_object_property(
            "hasStyle",
            domain="Product",
            range_="StyleEntity",
        )
        self._writer.add_object_property(
            "hasMaterial",
            domain="Product",
            range_="MaterialEntity",
        )
        self._writer.add_object_property(
            "hasFunction",
            domain="Product",
            range_="FunctionEntity",
        )
        self._writer.add_object_property(
            "hasColor",
            domain="Product",
            range_="ColorEntity",
        )
        self._writer.add_object_property(
            "belongsTo",
            domain="Product",
            range_="ProductConceptualDesignKnowledge",
            is_transitive=True,
            comment="Taxonomic membership (transitive).",
        )
        self._writer.add_object_property(
            "relatedTo",
            domain="ProductConceptualDesignKnowledge",
            range_="ProductConceptualDesignKnowledge",
            is_symmetric=True,
        )

        # ── Geo / business relations (Table 7)
        self._writer.add_object_property(
            "locatedInProvince",
            domain="Customer",
            range_="Province",
        )
        self._writer.add_object_property(
            "locatedInCity",
            domain="Customer",
            range_="City",
        )
        self._writer.add_object_property(
            "installsIn",
            domain="Product",
            range_="FunctionPart",
        )
        self._writer.add_object_property(
            "rangeOf",
            domain="Product",
            range_="Product",
            is_symmetric=True,
        )

    def _define_data_properties(self) -> None:
        """Define data properties (literal attributes)."""
        props = [
            ("hasName", "ProductConceptualDesignKnowledge", "string"),
            ("hasDescription", "ProductConceptualDesignKnowledge", "string"),
            ("hasPrice", "Product", "decimal"),
            ("hasDimension", "Product", "string"),
            ("hasWeight", "Product", "decimal"),
            ("hasManufacturer", "Product", "string"),
            ("hasModelNumber", "Product", "string"),
            ("hasYear", "ProductionProcessKnowledge", "integer"),
            ("hasCertification", "Product", "string"),
            ("hasImageURL", "ProductConceptualDesignKnowledge", "anyURI"),
        ]
        for name, domain, range_xsd in props:
            self._writer.add_data_property(name, domain=domain, range_xsd=range_xsd)

    def build(self) -> str:
        """Build and serialize the ontology to OWL/RDF-XML."""
        print("[OntologyBuilder] Defining class taxonomy...")
        self._define_classes()
        print("[OntologyBuilder] Defining object properties...")
        self._define_object_properties()
        print("[OntologyBuilder] Defining data properties...")
        self._define_data_properties()
        print(f"[OntologyBuilder] Serializing to {self.output_path} ...")
        self._writer.serialize(self.output_path)
        print("[OntologyBuilder] Done.")
        return self.output_path

    def add_individual_from_kb(
        self,
        term: str,
        class_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a knowledge base term as an OWL named individual."""
        self._writer.add_individual(
            name=term.replace(" ", "_"),
            class_name=class_name,
            properties=properties or {"hasName": term},
        )

    def populate_from_knowledge_base(
        self,
        kb_dict: Dict[str, Dict[str, Any]],
        class_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Populate ontology individuals from the knowledge base.

        Args:
            kb_dict:       {term: {tag, score, source, ...}}
            class_mapping: {tag_name: OWL_class_name}
        """
        default_class_map = {
            "客厅装修": "SpatialStructure",
            "卧室装修": "SpatialStructure",
            "木材": "MaterialEntity",
            "大理石": "MaterialEntity",
            "北欧风格": "StyleEntity",
            "现代简约": "StyleEntity",
            "沙发": "FurnitureProduct",
            "冰箱": "ApplianceProduct",
        }
        mapping = class_mapping or default_class_map

        for term, meta in kb_dict.items():
            tag = meta.get("tag", "")
            class_name = mapping.get(tag, "ProductConceptualDesignKnowledge")
            self.add_individual_from_kb(term, class_name, {"hasName": term})

        self._writer.serialize(self.output_path)
        print(
            f"[OntologyBuilder] Populated {len(kb_dict)} individuals "
            f"→ {self.output_path}"
        )
