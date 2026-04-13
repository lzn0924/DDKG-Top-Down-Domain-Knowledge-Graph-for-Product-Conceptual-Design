# DDKG: Top-Down Domain Knowledge Graph for Product Conceptual Design

Implementation code for the paper published in *Journal of Mechanical Design* (2025).

> Li Z, Sa G*, Liu Z, Li B, & Tan J. (2025). Top-Down Hierarchical Construction and Application of a Domain Knowledge Graph Based on Multimodal Design Information. *Journal of Mechanical Design*, 147(3): 031401.

---

## Data Availability

**The training data and annotation files are not included in this repository.**
The domain corpus was collected from proprietary industrial databases and is subject to a confidentiality agreement with the collaborating enterprise. If you need access for research purposes, please contact the corresponding author (Sa G).

A placeholder `data/` directory with format descriptions is provided so the pipeline can be configured and tested with your own data.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- Neo4j ≥ 4.x (for graph storage stage)
- THULAC ([installation guide](https://github.com/thunlp/THULAC-Python))

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
Top-down/
├── config.py                        # Hyperparameters and path configuration
├── main.py                          # Pipeline entry point
│
├── data/
│   ├── data_collection.py           # Web crawler (Scrapy)
│   └── data_preprocessing.py        # Text cleaning, dataset splitting
│
├── ontology/
│   ├── terminology_extraction.py    # Tag-based clustering (TF-IDF + Word2Vec)
│   ├── incremental_annotation.py    # Semi-supervised annotation engine
│   └── ontology_builder.py          # OWL/RDF ontology construction
│
├── knowledge_extraction/
│   ├── text_processor.py            # THULAC segmentation and POS tagging
│   ├── ner_model.py                 # LEBERT + BiLSTM-Attention-CRF
│   ├── relation_extractor.py        # Rule-based relation extraction
│   └── entity_linking.py            # Entity linking and coreference resolution
│
├── knowledge_migration/
│   ├── structured_mapping.py        # Relational DB → RDF triples (R2RML-style)
│   └── unstructured_mapping.py      # Text/image → triples
│
├── knowledge_graph/
│   └── neo4j_manager.py             # Neo4j CRUD, Cypher queries, bulk import
│
├── knowledge_application/
│   ├── qa_model.py                  # Joint NER + intent classification model
│   ├── similarity_model.py          # Contrastive learning for semantic similarity
│   └── knowledge_service.py         # Search, recommendation, Q&A service layer
│
└── evaluation/
    ├── evaluate_ner.py              # Token-level and span-level NER metrics
    ├── evaluate_similarity.py       # Spearman's ρ on STS-B format pairs
    └── evaluate_qa.py               # Weighted F1 for intent classification
```

---

## Usage

Before running, set `NEO4J_CONFIG` credentials and model paths in `config.py`.

```bash
# Ontology construction (terminology extraction + OWL build)
python main.py --stage ontology

# Knowledge extraction (NER + RE on a text corpus)
python main.py --stage extract

# Train the NER model
python main.py --stage train_ner --train_ner_data path/to/train.conll

# Train the joint Q&A model
python main.py --stage train_qa --train_qa_data path/to/qa_train.json

# Train the contrastive similarity model
python main.py --stage train_similarity --sim_corpus path/to/sentences.txt

# Import triples into Neo4j
python main.py --stage neo4j

# Run evaluation (prints comparison tables)
python main.py --stage evaluate

# Full pipeline
python main.py --no_crawl
```

---

## Citation

```bibtex
@article{li2025top,
  title     = {Top-Down Hierarchical Construction and Application of a Domain
               Knowledge Graph Based on Multimodal Design Information},
  author    = {Li, Zhinan and Sa, Guohua and Liu, Zhifeng and Li, Bo and Tan, Jianrong},
  journal   = {Journal of Mechanical Design},
  volume    = {147},
  number    = {3},
  pages     = {031401},
  year      = {2025},
  publisher = {American Society of Mechanical Engineers}
}
```
