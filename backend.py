#!/usr/bin/env python3
"""
Image Metadata and Tagging System - Backend Functions

This module contains all the core functionality for image metadata extraction,
tagging, speech bubble detection, and file tracking.
"""

# Imports
# ------------------------------
import argparse
import base64
import cv2
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytesseract
import torch
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForImageClassification
from typing import Dict, List, Set, Optional, Tuple

# Global Variables
# ------------------------------
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_FILE = "image_metadata.json"
TOOL_VERSION = "v0.1.0"
TAG_SYNONYM_MAP = {}

# Default tag categories
DEFAULT_TAG_CATEGORIES = {
    "hair_color": ["blonde_hair", "brown_hair", "black_hair", "red_hair", "white_hair"],
    "expression": ["smile", "serious", "angry", "sad", "surprised"],
    "clothing": ["school_uniform", "dress", "shirt", "jacket"],
    "body_parts": ["long_hair", "short_hair", "blue_eyes", "brown_eyes"],
    "background": ["indoors", "outdoors", "simple_background"],
    "uncategorized": []
}

# Classes
# ------------------------------
class FileTracker:
    def __init__(self, metadata_path: str = "image_metadata.json"):
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, dict]:
        """Load existing metadata or create new metadata file if none exists."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {self.metadata_path}, creating new metadata")
                return {}
        else:
            return {}
    
    def save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def update_file_status(self, base_dirs: List[str], extensions: Set[str] = SUPPORTED_EXTENSIONS):
        """Update file status flags and identify deleted files."""
        existing_files = set()
        
        for base_dir in base_dirs:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        full_path = os.path.normpath(os.path.join(root, file))
                        existing_files.add(full_path)
                        
                        if full_path in self.metadata:
                            self.metadata[full_path]["exists"] = True
                        else:
                            self.metadata[full_path] = {
                                "exists": True,
                                "last_modified": os.path.getmtime(full_path),
                                "size": os.path.getsize(full_path)
                            }
        
        for path in list(self.metadata.keys()):
            if path not in existing_files:
                self.metadata[path]["exists"] = False
        
        self.save_metadata()
        
        return {
            "total_files": len(self.metadata),
            "existing_files": len(existing_files),
            "deleted_files": len(self.metadata) - len(existing_files)
        }
    
    def find_duplicates(self, check_method: str = "path") -> Dict[str, List[str]]:
        """Find potential duplicate images using different methods."""
        duplicates = {}
        
        if check_method == "path":
            by_name = {}
            for path in self.metadata:
                filename = os.path.basename(path)
                by_name.setdefault(filename, []).append(path)
            duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
            
        elif check_method == "size":
            by_size = {}
            for path, info in self.metadata.items():
                if info.get("exists", True) and "size" in info:
                    size = info["size"]
                    by_size.setdefault(size, []).append(path)
            duplicates = {f"size_{size}": paths for size, paths in by_size.items() if len(paths) > 1}
            
        elif check_method == "metadata":
            by_meta = {}
            for path, info in self.metadata.items():
                if info.get("exists", True):
                    resolution = info.get("resolution", "")
                    img_format = info.get("format", "")
                    seed = info.get("seed", "")
                    fingerprint = f"{resolution}_{img_format}_{seed}"
                    by_meta.setdefault(fingerprint, []).append(path)
            duplicates = {meta: paths for meta, paths in by_meta.items() 
                         if len(paths) > 1 and meta != "__"}
        
        return duplicates

# Utility Functions
# ------------------------------
def hash_file(filepath: str) -> str:
    """Generate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_images(base_dir: str, extensions: List[str] = None) -> List[str]:
    """Find all image files in a directory and subdirectories."""
    if extensions is None:
        extensions = list(SUPPORTED_EXTENSIONS)
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(base_dir).rglob(f"*{ext}"))
    return [str(p.resolve()) for p in image_paths]

def load_metadata(metadata_path="image_metadata.json") -> Dict[str, dict]:
    """Load metadata from JSON file."""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(data, metadata_path="image_metadata.json"):
    """Save metadata to JSON file."""
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_image_metadata(image_path: str) -> Dict:
    """Extract comprehensive metadata from an image file."""
    metadata = {
        "file_name": os.path.basename(image_path),
        "folder": str(Path(image_path).parent),
        "tool_version": TOOL_VERSION,
        "tags": [],
        "source_tags": {},
        "predicted_tags": {},
        "seed": None,
        "steps": None,
        "sampler": None,
        "resolution": None,
        "lora_used": [],
        "encoder": None,
    }
    
    try:
        with Image.open(image_path) as img:
            metadata["resolution"] = f"{img.width}x{img.height}"
            metadata["format"] = img.format
            
            # Extract EXIF data
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[f"exif_{tag}"] = str(value)
            
            # Extract generation parameters from PNG info
            if hasattr(img, 'text') and img.text:
                for key, value in img.text.items():
                    if key.lower() in ['parameters', 'prompt', 'negative prompt']:
                        metadata[key.lower()] = value
                        
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
    
    return metadata

def scan_and_update_images(base_dir: str = ".") -> Dict[str, dict]:
    """Scan directory for images and update metadata."""
    images = find_images(base_dir)
    metadata = load_metadata()
    updated = False
    
    for img_path in images:
        str_path = str(Path(img_path).resolve())
        if str_path not in metadata:
            metadata[str_path] = extract_image_metadata(img_path)
            updated = True
    
    if updated:
        save_metadata(metadata)
    return metadata

def detect_speech_bubbles(image_path: str, output_path: str = None, draw_bounding: bool = True) -> List[Dict]:
    """Detect speech bubbles in an image using computer vision techniques."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    speech_bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for speech bubbles
                speech_bubbles.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })
                
                if draw_bounding:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if output_path and draw_bounding:
        cv2.imwrite(output_path, img)
    
    return speech_bubbles

def overwrite_speech_bubbles(image_path: str, output_path: str, bubble_regions: List[Dict] = None) -> None:
    """Overwrite detected speech bubbles with white rectangles."""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    if bubble_regions is None:
        bubble_regions = detect_speech_bubbles(image_path, draw_bounding=False)
    
    for bubble in bubble_regions:
        x, y, w, h = bubble["x"], bubble["y"], bubble["width"], bubble["height"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    
    cv2.imwrite(output_path, img)

def multi_prompt_difference(metadata: Dict[str, dict], base_prompt: str) -> Dict[str, List[str]]:
    """Compare prompts across images to find differences from a base prompt."""
    differences = {}
    base_words = set(base_prompt.lower().split())
    
    for image_path, meta in metadata.items():
        prompt = meta.get("parameters", "")
        if prompt:
            prompt_words = set(prompt.lower().split())
            diff_words = prompt_words - base_words
            if diff_words:
                differences[image_path] = list(diff_words)
    
    return differences

def compare_tag_sets(predicted: Dict[str, float], human: List[str], conf_threshold=0.2):
    """Compare predicted tags with human-annotated tags."""
    predicted_set = {tag for tag, conf in predicted.items() if conf >= conf_threshold}
    human_set = set(human)
    
    true_positives = predicted_set & human_set
    false_positives = predicted_set - human_set
    false_negatives = human_set - predicted_set
    
    precision = len(true_positives) / (len(predicted_set) or 1)
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
    """Evaluate all images with human annotations."""
    results = {}
    data = load_metadata(metadata_path)
    for path, meta in data.items():
        if "human_tags" in meta:
            predicted = meta.get("predicted_tags", {})
            human = meta.get("human_tags", [])
            results[path] = compare_tag_sets(predicted, human, conf_threshold)
    return results

def categorize_tags(tag_list: List[str], external_api_url: Optional[str] = None) -> Dict[str, str]:
    """Categorize a list of tags using either a lookup table or an external API."""
    if external_api_url:
        # Placeholder for external API call
        pass
    
    result = {}
    for tag in tag_list:
        categorized = False
        for category, tags in DEFAULT_TAG_CATEGORIES.items():
            if tag in tags:
                result[tag] = category
                categorized = True
                break
        if not categorized:
            result[tag] = "uncategorized"
    
    return result

def update_metadata_with_categories(metadata_path: str = "image_metadata.json", 
                                   output_path: str = "categorized_metadata.json",
                                   tag_key: str = "predicted_tags",
                                   external_api_url: Optional[str] = None) -> Dict[str, int]:
    """Update metadata file with tag categories."""
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found")
        return {}
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Collect all unique tags
    all_tags = set()
    for _, meta in metadata.items():
        if tag_key in meta:
            all_tags.update(meta[tag_key].keys())
    
    # Categorize all tags
    tag_categories = categorize_tags(list(all_tags), external_api_url)
    
    # Update metadata with categories
    category_counts = {}
    for image_path, meta in metadata.items():
        if tag_key in meta:
            meta["tag_categories"] = {}
            for tag in meta[tag_key]:
                category = tag_categories.get(tag, "uncategorized")
                meta["tag_categories"][tag] = category
                category_counts[category] = category_counts.get(category, 0) + 1
    
    # Save updated metadata
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Added tag categories to {len(metadata)} images in {output_path}")
    return category_counts

def group_by_tag(data, key="predicted_tags", min_conf=0.3):
    """Group images by tags above confidence threshold."""
    tag_groups = {}
    for image, info in data.items():
        tags = info.get(key, {})
        for tag, conf in tags.items():
            if conf >= min_conf:
                tag_groups.setdefault(tag, []).append(image)
    return tag_groups

# CLI Functions
# ------------------------------
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Image Metadata and Tagging Tool")
    parser.add_argument("--scan", action="store_true", help="Scan directory for images")
    parser.add_argument("--detect-bubbles", help="Detect speech bubbles in image")
    parser.add_argument("--categorize", action="store_true", help="Categorize tags")
    parser.add_argument("--base-dir", default=".", help="Base directory to scan")
    
    args = parser.parse_args()
    
    if args.scan:
        print("Scanning for images...")
        metadata = scan_and_update_images(args.base_dir)
        print(f"Found {len(metadata)} images")
    
    if args.detect_bubbles:
        bubbles = detect_speech_bubbles(args.detect_bubbles)
        print(f"Found {len(bubbles)} speech bubbles")
    
    if args.categorize:
        stats = update_metadata_with_categories()
        print(f"Categorization complete: {stats}")

if __name__ == "__main__":
    main()