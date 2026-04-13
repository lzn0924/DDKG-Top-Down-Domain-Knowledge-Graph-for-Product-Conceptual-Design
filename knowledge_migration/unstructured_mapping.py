"""
Unstructured data mapping: Text and Image → Knowledge Graph triples.

Implements the Non-Structured Data Mapping Technique (Section 2.2.2):
  "NLP techniques to map vocabularies, entities, and semantic triples
   contained in text fields to elements in the knowledge graph."
  "Image data can also extract semantics through image classification,
   object recognition techniques and organize them into the graph."

Pipeline for text:
  raw text → THULAC segmentation → NER (LEBERT+BiLSTM-CRF) →
  Relation Extraction (rule-based) → RDF triples

Pipeline for images:
  image → ResNet/EfficientNet classification →
  object detection → label mapping → RDF triples

Paper: Li Z et al. (2025), JMD 147(3): 031401 – Section 2.2.2.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR, ONTOLOGY_CONFIG

NS = ONTOLOGY_CONFIG["namespace"]


# ---------------------------------------------------------------------------
# Text-to-triple mapping
# ---------------------------------------------------------------------------

class TextTripleMapper:
    """
    Extracts RDF triples from unstructured text using the NLP pipeline:
      THULAC segmentation → NER → Rule-based RE → entity linking → triples
    """

    def __init__(
        self,
        text_processor=None,
        ner_model=None,
        relation_extractor=None,
        entity_linker=None,
    ):
        """
        Args:
            text_processor:    ChineseTextProcessor instance.
            ner_model:         LEBERTBiLSTMAttentionCRF instance (inference mode).
            relation_extractor: RuleBasedRelationExtractor instance.
            entity_linker:     EntityLinker instance.

        Note: All components are optional; pass None to skip that step.
        """
        self.processor = text_processor
        self.ner = ner_model
        self.re = relation_extractor
        self.linker = entity_linker

    def text_to_triples(
        self,
        text: str,
        source_uri: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Convert a text document to RDF triples.

        Returns:
            List of (subject_uri, property_uri, object_uri_or_literal) tuples.
        """
        triples = []

        # 1. Segmentation + noun phrase extraction
        if self.processor:
            noun_phrases = self.processor.extract_noun_phrases(text)
        else:
            noun_phrases = []

        # 2. NER (if model available; otherwise use noun phrases as entities)
        entities = []
        if self.ner is not None:
            pass   # inference call handled externally; pass pre-computed entities
        else:
            from knowledge_extraction.relation_extractor import Entity
            for np_ in noun_phrases[:20]:    # Limit candidate set
                entities.append(Entity(text=np_, label="ENTITY", start=-1, end=-1))

        # 3. Relation extraction
        if self.re is not None:
            relations = self.re.extract(text, entities)
            for rel in relations:
                head_uri = f"{NS}{rel.head.text.replace(' ', '_')}"
                tail_uri = f"{NS}{rel.tail.text.replace(' ', '_')}"
                prop_uri = f"{NS}{rel.relation}"
                triples.append((head_uri, prop_uri, tail_uri))

        # 4. Entity type assertions (rdf:type)
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        type_map = {
            "MATERIAL": f"{NS}MaterialEntity",
            "STYLE": f"{NS}StyleEntity",
            "SPACE": f"{NS}SpatialStructure",
            "PRODUCT": f"{NS}Product",
            "FUNCTION": f"{NS}FunctionEntity",
            "COLOR": f"{NS}ColorEntity",
        }
        for ent in entities:
            entity_uri = f"{NS}{ent.text.replace(' ', '_')}"
            class_uri = type_map.get(ent.label, f"{NS}ProductConceptualDesignKnowledge")
            triples.append((entity_uri, rdf_type, class_uri))

        # 5. Provenance triple (source document)
        if source_uri and triples:
            prov_prop = f"{NS}extractedFrom"
            for subj, _, _ in triples[:1]:
                triples.append((subj, prov_prop, source_uri))

        return triples

    def batch_text_to_triples(
        self,
        records: List[Dict[str, Any]],
        text_field: str = "content",
    ) -> List[Tuple[str, str, str]]:
        """Process a list of records and collect all triples."""
        all_triples = []
        for rec in records:
            text = rec.get(text_field, "")
            source = rec.get("url", "")
            triples = self.text_to_triples(text, source_uri=source)
            all_triples.extend(triples)
        return all_triples

    def write_ntriples(
        self,
        triples: List[Tuple[str, str, str]],
        output_path: str,
    ) -> int:
        """Write triples to N-Triples file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        count = 0
        with open(output_path, "a", encoding="utf-8") as f:
            for subj, pred, obj in triples:
                if obj.startswith("http"):
                    line = f"<{subj}> <{pred}> <{obj}> .\n"
                else:
                    escaped = obj.replace('"', '\\"')
                    line = f'<{subj}> <{pred}> "{escaped}"@zh .\n'
                f.write(line)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Image-to-triple mapping
# ---------------------------------------------------------------------------

class ImageTripleMapper:
    """
    Extracts design knowledge from product images via:
      - Image classification (style, material category)
      - Object detection (furniture, appliance identification)

    Uses torchvision models pretrained on ImageNet, fine-tuned on domain data.
    """

    # Domain label taxonomy for home furnishing images
    STYLE_LABELS = [
        "nordic", "modern_minimalist", "chinese_traditional",
        "american", "european_luxury", "japanese", "industrial",
        "mediterranean", "pastoral", "mixed",
    ]
    PRODUCT_LABELS = [
        "sofa", "bed", "wardrobe", "dining_table", "chair",
        "coffee_table", "tv_cabinet", "bookshelf", "desk",
        "refrigerator", "washing_machine", "air_conditioner",
    ]
    MATERIAL_LABELS = [
        "solid_wood", "marble", "glass", "metal", "fabric",
        "leather", "rattan", "bamboo", "stone", "ceramic",
    ]

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.classifier = self._load_classifier(model_path)

    def _load_classifier(self, model_path: Optional[str]):
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T

        if model_path and os.path.exists(model_path):
            model = torch.load(model_path, map_location=self.device)
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            total_labels = (
                len(self.STYLE_LABELS)
                + len(self.PRODUCT_LABELS)
                + len(self.MATERIAL_LABELS)
            )
            model.fc = nn.Linear(model.fc.in_features, total_labels)

        model = model.to(self.device)
        model.eval()

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        return model

    def classify_image(
        self, image_path: str
    ) -> Dict[str, Any]:
        """
        Classify a product image to extract style, product type, and material.

        Returns:
            Dict with keys: style, product_type, material, confidence.
        """
        if self.classifier is None:
            return {"style": None, "product_type": None, "material": None}

        import torch
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ImageMapper] Cannot open image {image_path}: {e}")
            return {}

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(tensor)[0]

        n_style = len(self.STYLE_LABELS)
        n_prod = len(self.PRODUCT_LABELS)

        style_logits = logits[:n_style]
        prod_logits = logits[n_style: n_style + n_prod]
        mat_logits = logits[n_style + n_prod:]

        style_idx = style_logits.argmax().item()
        prod_idx = prod_logits.argmax().item()
        mat_idx = mat_logits.argmax().item()

        return {
            "style": self.STYLE_LABELS[style_idx],
            "style_confidence": float(torch.softmax(style_logits, dim=0)[style_idx]),
            "product_type": self.PRODUCT_LABELS[prod_idx],
            "product_confidence": float(torch.softmax(prod_logits, dim=0)[prod_idx]),
            "material": self.MATERIAL_LABELS[mat_idx],
            "material_confidence": float(torch.softmax(mat_logits, dim=0)[mat_idx]),
        }

    def image_to_triples(
        self,
        image_path: str,
        product_uri: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Generate RDF triples from image classification results.

        Args:
            image_path:  Path to product image.
            product_uri: If known, the existing product entity URI.

        Returns:
            List of (subject, predicate, object) triples.
        """
        classifications = self.classify_image(image_path)
        if not classifications:
            return []

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        subj = product_uri or f"{NS}product_{image_name}"
        triples = []

        if classifications.get("style"):
            style_uri = f"{NS}style_{classifications['style']}"
            triples.append((subj, f"{NS}hasStyle", style_uri))
            triples.append((style_uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", f"{NS}StyleEntity"))

        if classifications.get("material"):
            mat_uri = f"{NS}material_{classifications['material']}"
            triples.append((subj, f"{NS}hasMaterial", mat_uri))
            triples.append((mat_uri, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", f"{NS}MaterialEntity"))

        if classifications.get("product_type"):
            triples.append((subj, f"{NS}hasProductType", classifications["product_type"]))

        triples.append((subj, f"{NS}hasImageURL", f"file://{image_path}"))
        return triples

    def batch_images_to_triples(
        self,
        image_dir: str,
        output_path: str,
        extensions: Tuple = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> int:
        """Process all images in a directory and write triples."""
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        all_triples = []
        for img_path in image_files:
            triples = self.image_to_triples(img_path)
            all_triples.extend(triples)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        count = 0
        with open(output_path, "a", encoding="utf-8") as f:
            for subj, pred, obj in all_triples:
                if obj.startswith("http") or obj.startswith(NS) or obj.startswith("file://"):
                    line = f"<{subj}> <{pred}> <{obj}> .\n"
                else:
                    escaped = obj.replace('"', '\\"')
                    line = f'<{subj}> <{pred}> "{escaped}" .\n'
                f.write(line)
                count += 1

        print(f"[ImageMapper] Generated {count} triples from {len(image_files)} images.")
        return count
