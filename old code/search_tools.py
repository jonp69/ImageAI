
# search_tools.py

import os
import json
from typing import List, Dict

def load_metadata(data_path="image_metadata.json") -> Dict[str, dict]:
    if not os.path.exists(data_path):
        return {}
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_by_tag(metadata: Dict[str, dict], tag: str, tag_type="predicted_tags", min_conf=0.3) -> List[str]:
    matched = []
    for path, info in metadata.items():
        tags = info.get(tag_type, {})
        if tag_type == "source_tags.positive":
            tags = info.get("source_tags", {}).get("positive", [])
            if tag in tags:
                matched.append(path)
        else:
            if tags.get(tag, 0) >= min_conf:
                matched.append(path)
    return matched

def find_similar_images(metadata: Dict[str, dict], target_tags: List[str], top_n: int = 10) -> List[str]:
    scores = []
    for path, info in metadata.items():
        tags = set(info.get("predicted_tags", {}).keys())
        score = len(set(target_tags) & tags)
        if score > 0:
            scores.append((path, score))
    scores.sort(key=lambda x: -x[1])
    return [path for path, _ in scores[:top_n]]

def multi_prompt_difference(metadata: Dict[str, dict], base_prompt: str) -> Dict[str, List[str]]:
    """
    Find regenerated images with slight variations of the same base prompt.
    """
    base = base_prompt.lower().strip()
    groups = {}
    for path, info in metadata.items():
        prompt = info.get("prompt", "").lower().strip()
        if base in prompt:
            groups.setdefault(prompt, []).append(path)
    return groups
