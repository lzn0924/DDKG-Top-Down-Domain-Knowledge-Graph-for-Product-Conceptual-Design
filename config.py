"""
Central configuration for DDKG (Design Domain Knowledge Graph).

All hyperparameters are sourced from:
  Li Z et al. (2025). "Top-Down Hierarchical Construction and Application of
  a Domain Knowledge Graph Based on Multimodal Design Information."
  Journal of Mechanical Design, 147(3): 031401.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_files")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for _d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# Pre-trained model identifiers (HuggingFace Hub or local path)
BERT_BASE_CHINESE = "bert-base-chinese"
BERT_WWM_EXT = "hfl/chinese-bert-wwm-ext"  # BERT-WWM used in contrastive model

# ---------------------------------------------------------------------------
# Data split ratios
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
VAL_RATIO = 0.10

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

TEXT_PROCESSOR = "thulac"  # Options: jieba | thulac | ltp | nlpir

# ---------------------------------------------------------------------------
# NER model: LEBERT + BiLSTM-Attention-CRF
# ---------------------------------------------------------------------------

NER_CONFIG = {
    "bert_model": BERT_BASE_CHINESE,
    "hidden_size": 768,
    "bilstm_hidden": 256,          # BiLSTM hidden units (per direction)
    "num_lstm_layers": 2,
    "attention_heads": 8,
    "dropout": 0.5,
    "learning_rate": 4e-5,
    "num_epochs": 50,
    "batch_size": 16,
    "max_seq_len": 128,
    "num_adversarial_per_sample": 5,
    "adversarial_loss_weight_intent": 0.25,
    "task_loss_weight_ner": 0.75,
    "crf": True,
    "lexicon_path": os.path.join(DATA_DIR, "lexicon.txt"),
}

# NER entity types (BIO scheme)
NER_LABELS = [
    "O",
    "B-PRODUCT", "I-PRODUCT",
    "B-MATERIAL", "I-MATERIAL",
    "B-STYLE", "I-STYLE",
    "B-SPACE", "I-SPACE",
    "B-FUNCTION", "I-FUNCTION",
    "B-BRAND", "I-BRAND",
    "B-COLOR", "I-COLOR",
    "B-PRICE", "I-PRICE",
]

# ---------------------------------------------------------------------------
# Query intent labels (24 categories)
# ---------------------------------------------------------------------------

INTENT_LABELS = [
    "design_style",
    "spatial_structure",
    "lighting_fixtures",
    "furniture",
    "appliance_configuration",
    "living_areas",
    "building_materials",
    "colors",
    "electrical_circuits",
    "decoration_methods",
    "structural_patterns",
    "parameter_configuration",
    "safety_requirements",
    "daily_maintenance",
    "facilities_for_disabled",
    "environmental_quality",
    "product_models",
    "soft_furnishings",
    "design_cases",
    "smart_home_technology",
    "ventilation_and_lighting",
    "protective_performance",
    "construction_notes",
    "living_facilities",
]

NUM_INTENT_CLASSES = len(INTENT_LABELS)  # 24

# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

RELATION_TYPES = [
    "usedIn",          # Material → Space
    "hasStyle",        # Product → Style
    "hasMaterial",     # Product → Material
    "hasFunction",     # Product → Function
    "hasColor",        # Product → Color
    "hasPrice",        # Product → Price
    "belongsTo",       # Entity → Category (taxonomy)
    "relatedTo",       # Generic association
    "mappingScheme",   # Scheme conception knowledge (reversible)
    "comparison",      # Scheme comparison (reversible)
    "dependency",      # Scheme → Design evaluation
    "composition",     # Scheme → Design object
    "attribute",       # Scheme → Design method
    "locatedIn",       # Customer → Province / City
    "installsIn",      # Product → Function part
    "rangeOf",         # Product → Product (range)
]

# ---------------------------------------------------------------------------
# Contrastive learning – semantic similarity
# ---------------------------------------------------------------------------

SIMILARITY_CONFIG = {
    "bert_model": BERT_WWM_EXT,
    "dropout": 0.3,
    "learning_rate": 1e-5,
    "num_epochs": 3,
    "batch_size": 16,
    "max_seq_len": 128,
    "temperature": 0.05,          # NT-Xent temperature τ
    "generic_samples": 5000,      # Sentences from Chinese-SNLI + Chinese-STS-B
    "domain_samples": 500,        # Home design domain corpus
    "val_samples": 120,           # STS-B format
    "test_samples": 120,          # STS-B format
    # Pooling strategies evaluated (P1–P4):
    # P1 = CLS vector of last encoder layer
    # P2 = BERT NSP pooler vector
    # P3 = mean of all vectors in last layer
    # P4 = mean of all vectors in first and last layers
    "pooling": "P4",              # Best single-model baseline; contrastive beats all
}

# ---------------------------------------------------------------------------
# Tag-based clustering
# ---------------------------------------------------------------------------

CLUSTERING_CONFIG = {
    "tfidf_min_df": 2,
    "tfidf_max_df": 0.95,
    "tfidf_threshold": 0.15,
    "word2vec_dim": 100,
    "word2vec_window": 5,
    "word2vec_min_count": 2,
    "similarity_threshold": 0.75, # Cosine similarity threshold for clustering
    "lexicon_coverage": 3,        # Number of lexicon expansions (sensitivity param)
    "initial_preset_tags": 50,    # Number of expert-preset seed tags
}

# ---------------------------------------------------------------------------
# Knowledge graph – Neo4j
# ---------------------------------------------------------------------------

NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "your_password_here",  # Replace with actual password
    "database": "ddkg",
}

# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

ONTOLOGY_CONFIG = {
    "namespace": "http://ddkg.design/ontology#",
    "output_path": os.path.join(DATA_DIR, "ddkg_ontology.owl"),
    "format": "rdfxml",
}

# Top-level concept classes
ONTOLOGY_CLASSES = [
    "ProductConceptualDesignKnowledge",
    "UserPreferenceKnowledge",
    "DesignObjectKnowledge",
    "DesignMethodKnowledge",
    "SchemeConceptionKnowledge",
    "DesignEvaluationKnowledge",
    "ProductionProcessKnowledge",
    "Product",
    "FurnitureProduct",
    "ApplianceProduct",
    "MaterialEntity",
    "StyleEntity",
    "SpatialStructure",
    "FunctionEntity",
    "DesignCase",
    "Customer",
    "Province",
    "City",
]
