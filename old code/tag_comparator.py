# tag_comparator.py

from typing import Dict, List

def compare_tags(generated: List[str], predicted: Dict[str, float], human: List[str] = None, threshold: float = 0.3):
    """
    Compare three sets of tags:
    - generated: Tags used in image generation
    - predicted: Tags predicted by the model with confidence values
    - human: Manually assigned tags (optional)

    Returns:
        dict with intersection, unique tags, and missing predictions
    """
    pred_tags = [tag for tag, conf in predicted.items() if conf >= threshold]
    gen_set = set(generated)
    pred_set = set(pred_tags)
    human_set = set(human or [])

    comparison = {
        "generated_only": list(gen_set - pred_set),
        "predicted_only": list(pred_set - gen_set),
        "common_tags": list(gen_set & pred_set),
        "missing_from_prediction": list(gen_set - pred_set),
        "human_discrepancies": None,
    }

    if human:
        comparison["human_discrepancies"] = {
            "missed_by_model": list(human_set - pred_set),
            "false_positives": list(pred_set - human_set),
            "correct_by_model": list(human_set & pred_set),
        }

    return comparison
