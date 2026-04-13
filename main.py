"""
Main entry point for the DDKG pipeline.

Orchestrates the full top-down hierarchical construction and application
of the Design Domain Knowledge Graph (DDKG) described in:

  Li Z, Sa G*, Liu Z, Li B, & Tan J. (2025).
  "Top-Down Hierarchical Construction and Application of a Domain
   Knowledge Graph Based on Multimodal Design Information."
  Journal of Mechanical Design, 147(3): 031401.

──────────────────────────────────────────────────────────────────────────
Pipeline (following Fig. 2 / Fig. 4 architecture):

  Stage 1 – Data Acquisition
    1a. Web crawl (Scrapy) → raw JSONL

  Stage 2 – Knowledge Representation (Ontology)
    2a. Terminology extraction (TF-IDF + tag-based clustering)
    2b. Incremental annotation (semi-supervised, experts + auto)
    2c. OWL ontology construction (data layer + schema layer)

  Stage 3 – Knowledge Extraction
    3a. THULAC text segmentation + POS tagging
    3b. NER  – LEBERT + BiLSTM-Attention-CRF
    3c. RE   – rule-based relation extraction
    3d. Entity linking & knowledge fusion

  Stage 4 – Knowledge Migration
    4a. Structured DB → N-Triples (R2RML-style)
    4b. Unstructured text + images → N-Triples

  Stage 5 – Knowledge Graph Storage
    5a. Import N-Triples into Neo4j

  Stage 6 – Knowledge Application
    6a. Train joint Q&A model (NER + intent classification)
    6b. Train contrastive semantic similarity model
    6c. Start knowledge service (retrieval + Q&A)

  Stage 7 – Evaluation
    7a. NER evaluation  (Table 2, Table 10)
    7b. Similarity evaluation  (Table 11, Appendix C)
    7c. Q&A evaluation  (Table 10)

Usage:
  python main.py [--stage STAGE] [--config CONFIG] [--device DEVICE]

Examples:
  python main.py                          # Run full pipeline
  python main.py --stage ontology         # Run ontology stage only
  python main.py --stage train_qa         # Train Q&A model only
  python main.py --stage evaluate         # Run all evaluations
  python main.py --stage demo             # Interactive Q&A demo
"""

import argparse
import json
import os
import sys

import torch

from config import (
    BASE_DIR,
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    NER_CONFIG,
    SIMILARITY_CONFIG,
    NEO4J_CONFIG,
    ONTOLOGY_CONFIG,
    CLUSTERING_CONFIG,
    TRAIN_RATIO,
    TEST_RATIO,
    VAL_RATIO,
)


# ---------------------------------------------------------------------------
# Helper – directory layout
# ---------------------------------------------------------------------------

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRIPLES_DIR = os.path.join(DATA_DIR, "triples")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotation")
NER_MODEL_DIR = os.path.join(MODEL_DIR, "ner")
QA_MODEL_DIR = os.path.join(MODEL_DIR, "qa")
SIM_MODEL_DIR = os.path.join(MODEL_DIR, "similarity")

for _d in [RAW_DATA_DIR, PROCESSED_DIR, TRIPLES_DIR, ANNOTATION_DIR,
           NER_MODEL_DIR, QA_MODEL_DIR, SIM_MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1 – Data Acquisition
# ---------------------------------------------------------------------------

def stage_data_collection() -> None:
    """Run the Scrapy-based web crawler to collect raw domain data."""
    print("\n" + "=" * 60)
    print("STAGE 1 – Data Acquisition")
    print("=" * 60)

    from data.data_collection import run_crawler

    seed_urls = [
        "https://www.jia.com",
        "https://www.hao123.com/home",
        "https://www.tubatu.com",
    ]
    output_path = os.path.join(RAW_DATA_DIR, "raw_corpus.jsonl")
    print(f"[Main] Starting crawler → {output_path}")
    print(f"       Seed URLs: {len(seed_urls)}")
    run_crawler(start_urls=seed_urls)
    print(f"[Main] Crawl complete. Raw data: {output_path}")


# ---------------------------------------------------------------------------
# Stage 2 – Ontology Construction
# ---------------------------------------------------------------------------

def stage_ontology(simulate_annotation: bool = True) -> None:
    """Terminology extraction, incremental annotation, and OWL ontology build."""
    print("\n" + "=" * 60)
    print("STAGE 2 – Knowledge Representation (Ontology)")
    print("=" * 60)

    from ontology.terminology_extraction import TagBasedClusteringPipeline
    from ontology.incremental_annotation import IncrementalAnnotationEngine, KnowledgeBase
    from ontology.ontology_builder import DDKGOntologyBuilder

    # 2a. Load corpus
    raw_path = os.path.join(RAW_DATA_DIR, "raw_corpus.jsonl")
    if not os.path.exists(raw_path):
        print("[Main] Raw corpus not found. Using demo sentences.")
        corpus = [
            "北欧风格的客厅通常采用浅色木材和简约家具，搭配白色墙面和绿植装饰。",
            "大理石是厨房台面和卫生间墙面的理想材料，具有耐磨耐热特性。",
            "现代简约风格强调功能性和极简美学，常见于城市公寓设计。",
            "实木沙发框架配合布艺坐垫，适合北欧和中式混搭风格的客厅。",
            "LED灯具节能环保，适用于客厅、卧室和餐厅的照明设计。",
            "意大利进口瓷砖广泛应用于高端住宅卫生间和厨房装修。",
            "智能家居系统可实现灯光、温度和安防的一体化控制。",
            "软装搭配包括窗帘、地毯、抱枕和装饰画等软性装饰元素。",
        ] * 50
    else:
        with open(raw_path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        corpus = [r.get("content", "") for r in records if r.get("content")]
    print(f"[Main] Corpus size: {len(corpus)} sentences")

    # 2b. Tag-based clustering (terminology extraction)
    print("[Main] Running tag-based clustering pipeline...")
    seed_tags = [
        "客厅装修", "卧室装修", "厨房设计", "卫生间", "北欧风格",
        "现代简约", "中式风格", "欧式风格", "实木家具", "软装搭配",
    ]
    # Tokenize corpus with simple character n-gram split for pipeline
    tokenized_corpus = [list(s) for s in corpus]
    pipeline = TagBasedClusteringPipeline(config=CLUSTERING_CONFIG)
    clusters = pipeline.run(corpus, tokenized_corpus, seed_tags)
    # Build a flat score dict from TF-IDF extractor for annotation engine
    scores = {term: 0.80 for terms in clusters.values() for term in terms}
    print(f"[Main] Clusters found: {len(clusters)}")
    for tag, terms in list(clusters.items())[:3]:
        print(f"         '{tag}': {terms[:5]}")

    # 2c. Incremental annotation
    kb_path = os.path.join(DATA_DIR, "knowledge_base.json")
    engine = IncrementalAnnotationEngine(kb_save_path=kb_path)
    summary = engine.run_iteration(
        raw_clusters=clusters,
        term_scores=scores,
        simulate=simulate_annotation,
        review_output_dir=ANNOTATION_DIR,
    )
    print(f"[Main] KB size: {summary['kb_size']} | Manual: {summary['manual_ratio']*100:.1f}%")

    # 2d. OWL ontology
    print("[Main] Building OWL ontology...")
    builder = DDKGOntologyBuilder()
    owl_path = builder.build()
    builder.populate_from_knowledge_base(engine.kb.terms)
    print(f"[Main] Ontology saved → {owl_path}")


# ---------------------------------------------------------------------------
# Stage 3 – Knowledge Extraction
# ---------------------------------------------------------------------------

def stage_knowledge_extraction(corpus_path: str = None) -> str:
    """Run NER and relation extraction on the processed corpus."""
    print("\n" + "=" * 60)
    print("STAGE 3 – Knowledge Extraction")
    print("=" * 60)

    from knowledge_extraction.text_processor import ChineseTextProcessor
    from knowledge_extraction.relation_extractor import RuleBasedRelationExtractor

    processor = ChineseTextProcessor()
    extractor = RuleBasedRelationExtractor()

    demo_texts = [
        "实木地板用于客厅和卧室装修，具有天然纹理和耐用性。",
        "北欧风格沙发的材质为实木框架配合高回弹海绵和布艺。",
        "大理石台面安装在厨房岛台，搭配不锈钢水槽和嵌入式灶具。",
        "该方案A与方案B相关，均采用现代简约风格进行家居设计。",
        "客户位于北京市，偏好轻奢风格家具和进口软装配饰。",
    ]

    triples_path = os.path.join(TRIPLES_DIR, "extracted_triples.nt")
    all_triples = []
    print(f"[Main] Extracting knowledge from {len(demo_texts)} demo sentences...")

    for text in demo_texts:
        words = processor.segment(text)
        relations = extractor.extract(text)
        triples = extractor.to_triples(relations)
        all_triples.extend(triples)
        if relations:
            print(f"  Text: {text[:30]}...")
            for rel in relations[:2]:
                print(f"    ({rel.head.text}) –[{rel.relation}]→ ({rel.tail.text})")

    os.makedirs(TRIPLES_DIR, exist_ok=True)
    with open(triples_path, "w", encoding="utf-8") as f:
        NS = ONTOLOGY_CONFIG["namespace"]
        for h, r, t in all_triples:
            f.write(f"<{NS}{h.replace(' ', '_')}> <{NS}{r}> <{NS}{t.replace(' ', '_')}> .\n")

    print(f"[Main] Extracted {len(all_triples)} triples → {triples_path}")
    return triples_path


# ---------------------------------------------------------------------------
# Stage 4 – Knowledge Migration
# ---------------------------------------------------------------------------

def stage_knowledge_migration() -> str:
    """Map structured and unstructured data to RDF triples."""
    print("\n" + "=" * 60)
    print("STAGE 4 – Knowledge Migration")
    print("=" * 60)

    from knowledge_migration.structured_mapping import StructuredKnowledgeMigrator
    from knowledge_migration.unstructured_mapping import TextTripleMapper
    from knowledge_extraction.text_processor import ChineseTextProcessor
    from knowledge_extraction.relation_extractor import RuleBasedRelationExtractor

    output_path = os.path.join(TRIPLES_DIR, "migrated_triples.nt")
    migrator = StructuredKnowledgeMigrator()

    # Structured: demo records (mimicking relational DB rows)
    demo_products = [
        {"product_id": "1", "product_name": "北欧实木沙发", "description": "三人位布艺沙发",
         "price": "5999", "model_number": "SF-2024-A", "manufacturer": "原木家居",
         "material_id": "1", "style_id": "1"},
        {"product_id": "2", "product_name": "大理石餐桌", "description": "六人位进口大理石台面",
         "price": "12800", "model_number": "DT-2024-B", "manufacturer": "意达家居",
         "material_id": "2", "style_id": "2"},
    ]
    demo_materials = [
        {"material_id": "1", "material_name": "实木", "description": "天然实木，环保耐用"},
        {"material_id": "2", "material_name": "大理石", "description": "进口大理石，耐磨耐热"},
    ]
    demo_styles = [
        {"style_id": "1", "style_name": "北欧风格"},
        {"style_id": "2", "style_name": "现代简约"},
    ]

    total = 0
    for table_name, rows in [
        ("materials", demo_materials),
        ("styles", demo_styles),
        ("products", demo_products),
    ]:
        for triple in migrator.generate_triples_from_rows(table_name, rows):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(triple + "\n")
            total += 1

    # Unstructured: text-based triples
    processor = ChineseTextProcessor()
    re_model = RuleBasedRelationExtractor()
    text_mapper = TextTripleMapper(
        text_processor=processor,
        relation_extractor=re_model,
    )
    demo_texts = [
        {"content": "实木地板被用于客厅装修，搭配布艺窗帘和北欧风格装饰。",
         "url": "http://example.com/article/1"},
        {"content": "大理石台面应用于高端厨房设计，材质为意大利进口石材。",
         "url": "http://example.com/article/2"},
    ]
    text_triples = text_mapper.batch_text_to_triples(demo_texts)
    n_text = text_mapper.write_ntriples(text_triples, output_path)
    total += n_text

    print(f"[Main] Migration complete: {total} triples → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Stage 5 – Neo4j Import
# ---------------------------------------------------------------------------

def stage_neo4j_import(triples_path: str) -> None:
    """Import generated N-Triples into the Neo4j graph database."""
    print("\n" + "=" * 60)
    print("STAGE 5 – Knowledge Graph Storage (Neo4j)")
    print("=" * 60)

    from knowledge_graph.neo4j_manager import Neo4jManager

    try:
        with Neo4jManager(config=NEO4J_CONFIG) as manager:
            manager.initialize_schema()
            nodes, rels = manager.import_ntriples(triples_path)
            stats = manager.get_statistics()
            print(f"[Main] Graph stats: {stats['total_nodes']} nodes, "
                  f"{stats['total_relations']} relations")
    except (ConnectionError, ImportError) as e:
        print(f"[Main] Neo4j import skipped: {e}")
        print("       Set NEO4J_CONFIG in config.py to enable graph storage.")


# ---------------------------------------------------------------------------
# Stage 6a – Train NER model
# ---------------------------------------------------------------------------

def stage_train_ner(train_path: str = None, val_path: str = None) -> None:
    """Train the LEBERT + BiLSTM-Attention-CRF NER model."""
    print("\n" + "=" * 60)
    print("STAGE 6a – NER Model Training")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from knowledge_extraction.ner_model import (
        LEBERTBiLSTMAttentionCRF,
        NERDataset,
        NERTrainer,
    )
    from data.data_preprocessing import load_ner_data
    from config import NER_LABELS

    tokenizer = AutoTokenizer.from_pretrained(NER_CONFIG["bert_model"])
    label2id = {lbl: i for i, lbl in enumerate(NER_LABELS)}

    # Use provided paths or demo data
    if train_path and os.path.exists(train_path):
        train_data = load_ner_data(train_path)
    else:
        print("[Main] No train data provided. Using tiny demo set (2 samples).")
        train_data = [
            (list("北欧风格沙发的材质为实木"), ["B-STYLE", "I-STYLE", "I-STYLE", "I-STYLE",
                                                   "B-PRODUCT", "I-PRODUCT",
                                                   "O", "O", "O", "O",
                                                   "B-MATERIAL", "I-MATERIAL"]),
            (list("大理石台面用于厨房装修"), ["B-MATERIAL", "I-MATERIAL",
                                               "B-PRODUCT", "I-PRODUCT",
                                               "O", "O",
                                               "B-SPACE", "I-SPACE", "O"]),
        ]

    model = LEBERTBiLSTMAttentionCRF(config=NER_CONFIG)
    trainer = NERTrainer(model, config=NER_CONFIG)
    dataset = NERDataset(train_data, tokenizer, label2id)
    loader = DataLoader(dataset, batch_size=NER_CONFIG["batch_size"], shuffle=True)

    # Run 1 epoch for demo; set num_epochs in config for full training
    demo_config = dict(NER_CONFIG)
    demo_config["num_epochs"] = 1
    trainer.train(loader, num_epochs=1)
    trainer.save(NER_MODEL_DIR)


# ---------------------------------------------------------------------------
# Stage 6b – Train Joint Q&A model
# ---------------------------------------------------------------------------

def stage_train_qa(train_path: str = None) -> None:
    """Train the joint Q&A model (NER + intent classification)."""
    print("\n" + "=" * 60)
    print("STAGE 6b – Joint Q&A Model Training")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from knowledge_application.qa_model import JointQAModel, QADataset, QATrainer
    from config import INTENT_LABELS, NER_LABELS

    tokenizer = AutoTokenizer.from_pretrained(NER_CONFIG["bert_model"])
    label2id = {lbl: i for i, lbl in enumerate(NER_LABELS)}
    intent2id = {intent: i for i, intent in enumerate(INTENT_LABELS)}

    # Demo training data
    demo_data = [
        {"text": "客厅用什么风格的沙发好",
         "ner_labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
         "intent": "design_style"},
        {"text": "大理石台面多少钱",
         "ner_labels": ["B-MATERIAL", "I-MATERIAL", "B-PRODUCT", "I-PRODUCT", "O", "O", "O"],
         "intent": "building_materials"},
        {"text": "北欧风格卧室适合什么颜色",
         "ner_labels": ["B-STYLE", "I-STYLE", "I-STYLE", "I-STYLE",
                        "B-SPACE", "I-SPACE", "O", "O", "O", "O", "O"],
         "intent": "colors"},
        {"text": "客厅装修需要什么材料",
         "ner_labels": ["B-SPACE", "I-SPACE", "O", "O", "O", "O", "O", "O"],
         "intent": "building_materials"},
    ]

    if train_path and os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            demo_data = json.load(f)
        print(f"[Main] Loaded {len(demo_data)} QA training samples.")

    model = JointQAModel(config=NER_CONFIG)
    trainer = QATrainer(model, config=NER_CONFIG)
    dataset = QADataset(demo_data, tokenizer, label2id, intent2id)
    loader = DataLoader(dataset, batch_size=NER_CONFIG["batch_size"], shuffle=True)

    trainer.train(loader, num_epochs=1)
    trainer.save(QA_MODEL_DIR)


# ---------------------------------------------------------------------------
# Stage 6c – Train contrastive similarity model
# ---------------------------------------------------------------------------

def stage_train_similarity(corpus_path: str = None) -> None:
    """Train the unsupervised contrastive similarity model (Appendix C)."""
    print("\n" + "=" * 60)
    print("STAGE 6c – Contrastive Similarity Model Training")
    print("=" * 60)

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from knowledge_application.similarity_model import (
        ContrastiveSimilarityModel,
        UnsupervisedSimilarityDataset,
        SimilarityTrainer,
    )

    tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_CONFIG["bert_model"])

    # Demo corpus (mix generic + domain)
    demo_sentences = [
        "北欧风格的客厅通常采用浅色木材和简约家具。",
        "现代简约设计强调功能性和极简美学。",
        "大理石是厨房台面的理想材料，具有耐磨特性。",
        "实木地板适合客厅和卧室，纹理自然美观。",
        "LED灯具节能环保，适用于多种室内空间照明。",
        "智能家居系统实现灯光和温度一体化控制。",
        "软装搭配包括窗帘、地毯和装饰画等元素。",
        "意大利进口瓷砖广泛用于高端住宅卫生间装修。",
    ] * 64   # 512 sentences for demo

    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            demo_sentences = [l.strip() for l in f if l.strip()]
        print(f"[Main] Loaded {len(demo_sentences)} sentences from {corpus_path}")

    model = ContrastiveSimilarityModel(config=SIMILARITY_CONFIG)
    trainer = SimilarityTrainer(model, config=SIMILARITY_CONFIG)
    dataset = UnsupervisedSimilarityDataset(
        demo_sentences, tokenizer, max_length=SIMILARITY_CONFIG["max_seq_len"]
    )
    loader = DataLoader(dataset, batch_size=SIMILARITY_CONFIG["batch_size"], shuffle=True)

    trainer.train(loader, num_epochs=1)
    trainer.save(SIM_MODEL_DIR)


# ---------------------------------------------------------------------------
# Stage 7 – Evaluation
# ---------------------------------------------------------------------------

def stage_evaluate() -> None:
    """Run all evaluation scripts and print results tables."""
    print("\n" + "=" * 60)
    print("STAGE 7 – Evaluation")
    print("=" * 60)

    from evaluation.evaluate_ner import print_model_comparison_table as print_ner_table
    from evaluation.evaluate_qa import print_qa_comparison_table
    from evaluation.evaluate_similarity import run_full_ablation

    print_ner_table()
    print_qa_comparison_table()

    val_sts = os.path.join(PROCESSED_DIR, "val_sts.tsv")
    test_sts = os.path.join(PROCESSED_DIR, "test_sts.tsv")
    if os.path.exists(val_sts) and os.path.exists(test_sts):
        run_full_ablation(val_sts, test_sts)
    else:
        print("\n[Eval] STS evaluation files not found.")
        print(f"       Expected: {val_sts}")
        print("       Run stage_ontology + build_evaluation_datasets first.")
        print("\n[Table 11 – Paper Results]")
        rows = [
            ("BERT-base-Chinese P1", 0.7123, 0.7285),
            ("BERT-base-Chinese P2", 0.4367, 0.6132),
            ("BERT-base-Chinese P3", 0.7281, 0.7366),
            ("BERT-base-Chinese P4", 0.7509, 0.7683),
            ("BERT-WWM-ext P1",      0.4727, 0.4251),
            ("BERT-WWM-ext P2",      0.5012, 0.4219),
            ("BERT-WWM-ext P3",      0.6948, 0.6787),
            ("BERT-WWM-ext P4",      0.7012, 0.7285),
            ("Proposed (ours) ★",    0.8161, 0.8455),
        ]
        print(f"\n  {'Model':<30} {'Val ρ':>10} {'Test ρ':>10}")
        print("  " + "-" * 52)
        for name, val, test in rows:
            print(f"  {name:<30} {val:>10.4f} {test:>10.4f}")


# ---------------------------------------------------------------------------
# Stage – Interactive Q&A Demo
# ---------------------------------------------------------------------------

def stage_demo() -> None:
    """
    Interactive knowledge-graph Q&A demo.

    Prerequisites:
      1. Neo4j running and configured in config.py (NEO4J_CONFIG)
      2. QA model trained: python main.py --stage train_qa
      3. Similarity model trained: python main.py --stage train_similarity
    """
    print("\n" + "=" * 60)
    print("DDKG Interactive Q&A Demo (Section 3.2)")
    print("=" * 60)

    from transformers import AutoTokenizer
    from knowledge_application.qa_model import JointQAModel, QATrainer
    from knowledge_application.knowledge_service import (
        KnowledgeQAService,
        KnowledgeRetrievalService,
    )
    from knowledge_graph.neo4j_manager import Neo4jManager
    from config import NER_CONFIG

    # Load trained QA model
    qa_model = JointQAModel(config=NER_CONFIG)
    trainer = QATrainer(qa_model, device=device)
    trainer.load(QA_MODEL_DIR)

    # Connect Neo4j
    neo4j = Neo4jManager(config=NEO4J_CONFIG)
    neo4j.connect()

    qa_service = KnowledgeQAService(qa_model=qa_model, neo4j_manager=neo4j)

    example_queries = [
        "客厅用什么风格的沙发好看？",
        "大理石台面用什么材料？",
        "北欧风格卧室适合什么颜色搭配？",
        "智能家居系统如何安装？",
    ]

    print("\nExample queries:\n")
    for query in example_queries:
        result = qa_service.answer(query)
        print(f"  Q: {query}")
        print(f"     Entities : {result['entities']}")
        print(f"     Intent   : {result['intent']}")
        print(f"     Cypher   : {result['cypher'][:60]}...")
        print(f"     Answer   : {result['answer']}")
        print()

    neo4j.close()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDKG Pipeline – Top-Down Domain Knowledge Graph"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "collect",
            "ontology",
            "extract",
            "migrate",
            "neo4j",
            "train_ner",
            "train_qa",
            "train_similarity",
            "evaluate",
            "demo",
        ],
        help=(
            "Pipeline stage to run. "
            "'all' runs the complete pipeline end-to-end."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device: 'cuda' | 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--train_ner_data",
        type=str,
        default=None,
        help="Path to CoNLL-format NER training data.",
    )
    parser.add_argument(
        "--train_qa_data",
        type=str,
        default=None,
        help="Path to JSON QA training data.",
    )
    parser.add_argument(
        "--sim_corpus",
        type=str,
        default=None,
        help="Path to plain-text corpus for contrastive similarity training.",
    )
    parser.add_argument(
        "--no_crawl",
        action="store_true",
        help="Skip the web crawl stage even in 'all' mode.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DDKG] Device: {device}")
    print(f"[DDKG] Base dir: {BASE_DIR}")
    print(f"[DDKG] Stage: {args.stage}")

    stage = args.stage

    if stage in ("all", "collect"):
        if not args.no_crawl:
            stage_data_collection()
        else:
            print("[Main] Skipping data collection (--no_crawl).")

    if stage in ("all", "ontology"):
        stage_ontology(simulate_annotation=True)

    if stage in ("all", "extract"):
        triples_path = stage_knowledge_extraction()

    if stage in ("all", "migrate"):
        triples_path = stage_knowledge_migration()

    if stage in ("all", "neo4j"):
        merged_path = os.path.join(TRIPLES_DIR, "migrated_triples.nt")
        if os.path.exists(merged_path):
            stage_neo4j_import(merged_path)
        else:
            print("[Main] No triples file found. Run 'migrate' stage first.")

    if stage in ("all", "train_ner"):
        stage_train_ner(train_path=args.train_ner_data)

    if stage in ("all", "train_qa"):
        stage_train_qa(train_path=args.train_qa_data)

    if stage in ("all", "train_similarity"):
        stage_train_similarity(corpus_path=args.sim_corpus)

    if stage in ("all", "evaluate"):
        stage_evaluate()

    if stage == "demo":
        stage_demo()

    print("\n[DDKG] Done.")


if __name__ == "__main__":
    main()
