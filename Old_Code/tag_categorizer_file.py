# tag_categorizer.py

import json
import os
from typing import Dict, List, Optional

# Default categories for common tags
DEFAULT_CATEGORIES = {
    # Subject categories
    "1girl": "character",
    "1boy": "character",
    "female": "character",
    "male": "character",
    "person": "character",
    
    # Style categories
    "realistic": "style",
    "anime": "style",
    "digital_art": "style",
    "sketch": "style",
    "painting": "style",
    
    # Color categories
    "red_hair": "hair",
    "blonde_hair": "hair",
    "black_hair": "hair",
    "blue_hair": "hair",
    "green_hair": "hair",
    
    # Eye categories
    "blue_eyes": "eyes",
    "green_eyes": "eyes",
    "brown_eyes": "eyes",
    "red_eyes": "eyes",
    
    # Background categories
    "forest": "background",
    "city": "background",
    "night": "background",
    "indoors": "background",
    "outdoors": "background",
    
    # Clothing categories
    "dress": "clothing",
    "suit": "clothing",
    "uniform": "clothing",
    "hat": "clothing",
    "glasses": "clothing",
    
    # Expression categories
    "smile": "expression",
    "smiling": "expression",
    "frown": "expression",
    "crying": "expression",
    "laughing": "expression",
    
    # Medium categories
    "photo": "medium",
    "drawing": "medium",
    "3d_render": "medium",
    "digital_painting": "medium"
}

def categorize_tags(tag_list: List[str], external_api_url: Optional[str] = None) -> Dict[str, str]:
    """
    Categorize a list of tags using either a lookup table or an external API.
    
    This is a stub implementation that would normally connect to an external 
    LLM API to categorize tags, but for now uses a default mapping.
    
    Args:
        tag_list: List of tags to categorize
        external_api_url: Optional URL for external categorization API
        
    Returns:
        Dictionary mapping tags to their categories
    """
    # This would be where we'd call an external API/LLM
    if external_api_url:
        # Stub for external API call - in a real implementation, you would:
        # 1. Send the tag list to the external API
        # 2. Process the response
        # 3. Return the categorized tags
        print(f"Would call external API at {external_api_url} with {len(tag_list)} tags")
        
        # For now, fall back to default categories
        pass
    
    # Use default categories for known tags
    result = {}
    for tag in tag_list:
        if tag in DEFAULT_CATEGORIES:
            result[tag] = DEFAULT_CATEGORIES[tag]
        else:
            # For unknown tags, assign a default "other" category
            result[tag] = "other"
    
    return result

def update_metadata_with_categories(metadata_path: str = "image_metadata.json", 
                                   output_path: str = "categorized_metadata.json",
                                   tag_key: str = "predicted_tags",
                                   external_api_url: Optional[str] = None) -> Dict[str, int]:
    """
    Update metadata file with tag categories.
    
    Args:
        metadata_path: Path to input metadata JSON file
        output_path: Path to output the updated metadata
        tag_key: Key in metadata for the tags to categorize
        external_api_url: Optional URL for external categorization API
        
    Returns:
        Dictionary with statistics about the categorization
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata found at {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Collect all unique tags across all images
    all_tags = set()
    for _, meta in metadata.items():
        if tag_key in meta:
            all_tags.update(meta[tag_key].keys())
    
    # Categorize all tags at once
    tag_categories = categorize_tags(list(all_tags), external_api_url)
    
    # Update metadata with categories
    category_counts = {}
    for image_path, meta in metadata.items():
        if tag_key in meta:
            meta["tag_categories"] = {}
            for tag in meta[tag_key]:
                category = tag_categories.get(tag, "other")
                meta["tag_categories"][tag] = category
                category_counts[category] = category_counts.get(category, 0) + 1
    
    # Save updated metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Added tag categories to {len(metadata)} images in {output_path}")
    
    return category_counts