import json
import os
from typing import Dict, List

TAG_SYNONYM_MAP = {
    "redhead": "red_hair",
    "blonde": "blonde_hair",
    "blond_hair": "blonde_hair",
    "green eyes": "green_eyes",
    "blue eyes": "blue_eyes",
    "smile": "smiling",
    "open_mouth": "mouth_open",
    "closed_mouth": "mouth_closed",
    "girl": "female",
    "boy": "male"
}

def normalize_tag(tag: str) -> str:
    tag_lower = tag.strip().lower().replace(" ", "_")
    return TAG_SYNONYM_MAP.get(tag_lower, tag_lower)

def normalize_tag_list(tag_list: List[str]) -> List[str]:
    return list(set(normalize_tag(tag) for tag in tag_list))

def normalize_metadata_tags(metadata: Dict[str, dict], tag_key: str = "predicted_tags", threshold: float = 0.2):
    for image_path, meta in metadata.items():
        if tag_key not in meta:
            continue
        tag_dict = meta[tag_key]
        new_tags = {}
        for tag, score in tag_dict.items():
            if score >= threshold:
                norm = normalize_tag(tag)
                new_tags[norm] = max(new_tags.get(norm, 0), score)
        meta[tag_key] = new_tags
    return metadata

def normalize_metadata_file(input_path="image_metadata.json", output_path="normalized_metadata.json", tag_key="predicted_tags", threshold=0.2):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No metadata found at {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    normalized = normalize_metadata_tags(metadata, tag_key, threshold)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    print(f"[✓] Normalized tags saved to {output_path}")
