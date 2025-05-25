# auto_tagger.py

import os
from PIL import Image
from typing import Dict
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
import json

# Load pretrained WD tagger
def load_model():
    processor = AutoProcessor.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    model = AutoModelForImageClassification.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    return processor, model

# Predict tags with confidence
def predict_tags(image_path: str, processor, model, threshold=0.15) -> Dict[str, float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    probs = logits.sigmoid()
    tags = {}
    for tag, score in zip(model.config.id2label.values(), probs.tolist()):
        if score >= threshold:
            tags[tag] = score
    return tags

# Tag untagged images
def auto_tag_images(image_dir="images", metadata_path="image_metadata.json", formats={".jpg", ".jpeg", ".png", ".webp"}, conf_thresh=0.15):
    processor, model = load_model()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    for root, _, files in os.walk(image_dir):
        for file in files:
            if not any(file.lower().endswith(ext) for ext in formats):
                continue
            full_path = os.path.join(root, file)
            if full_path in metadata and "predicted_tags" in metadata[full_path]:
                continue  # Already tagged
            try:
                tags = predict_tags(full_path, processor, model, threshold=conf_thresh)
                metadata[full_path] = metadata.get(full_path, {})
                metadata[full_path]["predicted_tags"] = tags
                metadata[full_path]["version"] = "auto_tagger_v1"
            except Exception as e:
                print(f"Failed to tag {full_path}: {e}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)