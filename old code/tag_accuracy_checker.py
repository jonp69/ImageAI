# tag_accuracy_checker.py

from typing import Dict, List
import os
import json

def load_metadata(metadata_path="image_metadata.json") -> Dict[str, dict]:
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_tag_sets(predicted: Dict[str, float], human: List[str], conf_threshold=0.2):
    pred_set = {tag for tag, conf in predicted.items() if conf >= conf_threshold}
    human_set = set(human)

    true_positives = pred_set & human_set
    false_positives = pred_set - human_set
    false_negatives = human_set - pred_set

    precision = len(true_positives) / (len(pred_set) or 1)
    recall = len(true_positives) / (len(human_set) or 1)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def evaluate_all(metadata_path="image_metadata.json", conf_threshold=0.2):
    results = {}
    data = load_metadata(metadata_path)
    for path, meta in data.items():
        predicted = meta.get("predicted_tags", {})
        human = meta.get("source_tags", {}).get("human", [])
        if human and predicted:
            comparison = compare_tag_sets(predicted, human, conf_threshold)
            results[path] = comparison
    return results
