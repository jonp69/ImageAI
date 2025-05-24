#!/usr/bin/env python3
"""
Combined Python Script
Generated from: auto_tagger.py, auto_tagger_file.py, comic_splitter.py, comic_splitter_file.py, file_tracker(2).py, file_tracker.py, fixed_speech_bubble.py, image_tagging_cli.py, metadata_extractor.py, metadata_extractor_original.py, search_tools.py, speech_bubble_fixed(1).py, speech_bubble_fixed.py, tag_accuracy_checker.py, tag_categorizer.py, tag_categorizer_file(1).py, tag_categorizer_file.py, tag_comparator.py, tag_dashboard_tool (2).py, tag_dashboard_tool.py, tag_normalizer (2).py, tag_normalizer.py
Total files combined: 22
"""

# ==================================================
# Content from: auto_tagger.py
# ==================================================

# ==================================================
# Content from: auto_tagger_file.py
# ==================================================

# ==================================================
# Content from: comic_splitter.py
# ==================================================

# ==================================================
# Content from: comic_splitter_file.py
# ==================================================

# ==================================================
# Content from: file_tracker(2).py
# ==================================================

# ==================================================
# Content from: file_tracker.py
# ==================================================

# ==================================================
# Content from: fixed_speech_bubble.py
# ==================================================

# ==================================================
# Content from: image_tagging_cli.py
# ==================================================

# ==================================================
# Content from: metadata_extractor.py
# ==================================================

# ==================================================
# Content from: metadata_extractor_original.py
# ==================================================

# ==================================================
# Content from: search_tools.py
# ==================================================

# ==================================================
# Content from: speech_bubble_fixed(1).py
# ==================================================

# ==================================================
# Content from: speech_bubble_fixed.py
# ==================================================

# ==================================================
# Content from: tag_accuracy_checker.py
# ==================================================

# ==================================================
# Content from: tag_categorizer.py
# ==================================================

# ==================================================
# Content from: tag_categorizer_file(1).py
# ==================================================

# ==================================================
# Content from: tag_categorizer_file.py
# ==================================================

# ==================================================
# Content from: tag_comparator.py
# ==================================================

# ==================================================
# Content from: tag_dashboard_tool (2).py
# ==================================================

# ==================================================
# Content from: tag_dashboard_tool.py
# ==================================================

# ==================================================
# Content from: tag_normalizer (2).py
# ==================================================

# ==================================================
# Content from: tag_normalizer.py
# ==================================================

# Imports
# ------------------------------

import argparse
import base64
import cv2
import json
import matplotlib.pyplot
import numpy
import os
import pandas
import pytesseract
import seaborn
import streamlit
import torch
from PIL import Image
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tag_comparator import compare_tags
from tag_normalizer import normalize_metadata_file
from transformers import AutoProcessor, AutoModelForImageClassification
from typing import Dict
from typing import Dict, List
from typing import Dict, List, Optional
from typing import Dict, Set, List
from typing import List, Dict
from typing import List, Tuple

# Global Variables
# ------------------------------

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
metadata = load_metadata()
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_FILE = "image_metadata.json"
TOOL_VERSION = "v0.1.0"
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
    
    def update_file_status(self, base_dirs: List[str], extensions: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}):
        """
        Update file status flags:
        - Mark files as available (exists=True) if they exist
        - Mark files as deleted (exists=False) if they don't exist but are in metadata
        - Identify duplicates based on file hash or other properties
        
        Args:
            base_dirs: List of directories to scan for images
            extensions: Set of file extensions to consider
        """
        # Track existing files to identify what's been deleted
        existing_files = set()
        
        # Scan all directories for image files
        for base_dir in base_dirs:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        full_path = os.path.normpath(os.path.join(root, file))
                        existing_files.add(full_path)
                        
                        # Update or add metadata for this file
                        if full_path in self.metadata:
                            self.metadata[full_path]["exists"] = True
                        else:
                            # New file, add basic entry
                            self.metadata[full_path] = {
                                "exists": True,
                                "last_modified": os.path.getmtime(full_path),
                                "size": os.path.getsize(full_path)
                            }
        
        # Mark files that no longer exist as deleted, but keep their metadata
        for path in list(self.metadata.keys()):
            if path not in existing_files:
                self.metadata[path]["exists"] = False
        
        # Save updated metadata
        self.save_metadata()
        
        return {
            "total_files": len(self.metadata),
            "existing_files": len(existing_files),
            "deleted_files": len(self.metadata) - len(existing_files)
        }
    
    def find_duplicates(self, check_method: str = "path") -> Dict[str, List[str]]:
        """
        Find potential duplicate images using different methods.
        
        Args:
            check_method: Method to identify duplicates:
                - "path": Check if filename appears in multiple locations
                - "size": Group by identical file size
                - "metadata": Group by identical metadata parameters
                
        Returns:
            Dictionary of duplicate groups
        """
        duplicates = {}
        
        if check_method == "path":
            # Group by filename regardless of directory
            by_name = {}
            for path in self.metadata:
                filename = os.path.basename(path)
                by_name.setdefault(filename, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
            
        elif check_method == "size":
            # Group by file size
            by_size = {}
            for path, info in self.metadata.items():
                if info.get("exists", True) and "size" in info:
                    size = info["size"]
                    by_size.setdefault(size, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {f"size_{size}": paths for size, paths in by_size.items() if len(paths) > 1}
            
        elif check_method == "metadata":
            # Group by metadata fingerprint (resolution + format + seed if available)
            by_meta = {}
            for path, info in self.metadata.items():
                if info.get("exists", True):
                    # Create a metadata fingerprint
                    resolution = info.get("resolution", "")
                    img_format = info.get("format", "")
                    seed = info.get("seed", "")
                    fingerprint = f"{resolution}_{img_format}_{seed}"
                    
                    by_meta.setdefault(fingerprint, []).append(path)
            
            # Keep only groups with multiple entries and valid fingerprints
            duplicates = {meta: paths for meta, paths in by_meta.items() 
                         if len(paths) > 1 and meta != "__"}
        
        return duplicates

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
    
    def update_file_status(self, base_dirs: List[str], extensions: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}):
        """
        Update file status flags:
        - Mark files as available (exists=True) if they exist
        - Mark files as deleted (exists=False) if they don't exist but are in metadata
        - Identify duplicates based on file hash or other properties
        
        Args:
            base_dirs: List of directories to scan for images
            extensions: Set of file extensions to consider
        """
        # Track existing files to identify what's been deleted
        existing_files = set()
        
        # Scan all directories for image files
        for base_dir in base_dirs:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        full_path = os.path.normpath(os.path.join(root, file))
                        existing_files.add(full_path)
                        
                        # Update or add metadata for this file
                        if full_path in self.metadata:
                            self.metadata[full_path]["exists"] = True
                        else:
                            # New file, add basic entry
                            self.metadata[full_path] = {
                                "exists": True,
                                "last_modified": os.path.getmtime(full_path),
                                "size": os.path.getsize(full_path)
                            }
        
        # Mark files that no longer exist as deleted, but keep their metadata
        for path in list(self.metadata.keys()):
            if path not in existing_files:
                self.metadata[path]["exists"] = False
        
        # Save updated metadata
        self.save_metadata()
        
        return {
            "total_files": len(self.metadata),
            "existing_files": len(existing_files),
            "deleted_files": len(self.metadata) - len(existing_files)
        }
    
    def find_duplicates(self, check_method: str = "path") -> Dict[str, List[str]]:
        """
        Find potential duplicate images using different methods.
        
        Args:
            check_method: Method to identify duplicates:
                - "path": Check if filename appears in multiple locations
                - "size": Group by identical file size
                - "metadata": Group by identical metadata parameters
                
        Returns:
            Dictionary of duplicate groups
        """
        duplicates = {}
        
        if check_method == "path":
            # Group by filename regardless of directory
            by_name = {}
            for path in self.metadata:
                filename = os.path.basename(path)
                by_name.setdefault(filename, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
            
        elif check_method == "size":
            # Group by file size
            by_size = {}
            for path, info in self.metadata.items():
                if info.get("exists", True) and "size" in info:
                    size = info["size"]
                    by_size.setdefault(size, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {f"size_{size}": paths for size, paths in by_size.items() if len(paths) > 1}
            
        elif check_method == "metadata":
            # Group by metadata fingerprint (resolution + format + seed if available)
            by_meta = {}
            for path, info in self.metadata.items():
                if info.get("exists", True):
                    # Create a metadata fingerprint
                    resolution = info.get("resolution", "")
                    img_format = info.get("format", "")
                    seed = info.get("seed", "")
                    fingerprint = f"{resolution}_{img_format}_{seed}"
                    
                    by_meta.setdefault(fingerprint, []).append(path)
            
            # Keep only groups with multiple entries and valid fingerprints
            duplicates = {meta: paths for meta, paths in by_meta.items() 
                         if len(paths) > 1 and meta != "__"}
        
        return duplicates

# Functions
# ------------------------------

def load_model():
    processor = AutoProcessor.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    model = AutoModelForImageClassification.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    return processor, model

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

def load_model():
    processor = AutoProcessor.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    model = AutoModelForImageClassification.from_pretrained("SmilingWolf/wd-v1-4-convnextv2-tagger-v2")
    return processor, model

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

def split_panels(image_path: str, output_dir: str, remove_text: bool = False) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_paths = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 10000:
            continue  # skip small regions

        panel = image[y:y+h, x:x+w]

        if remove_text:
            panel = remove_text_from_image(panel)

        panel_filename = f"{basename}_panel_{i}.png"
        panel_path = os.path.join(output_dir, panel_filename)
        cv2.imwrite(panel_path, panel)
        panel_paths.append(panel_path)

    return panel_paths

def remove_text_from_image(image_np: np.ndarray) -> np.ndarray:
    d = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if w > 0 and h > 0:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return image_np

def split_panels(image_path: str, output_dir: str, remove_text: bool = False) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_paths = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 10000:
            continue  # skip small regions

        panel = image[y:y+h, x:x+w]

        if remove_text:
            panel = remove_text_from_image(panel)

        panel_filename = f"{basename}_panel_{i}.png"
        panel_path = os.path.join(output_dir, panel_filename)
        cv2.imwrite(panel_path, panel)
        panel_paths.append(panel_path)

    return panel_paths

def remove_text_from_image(image_np: np.ndarray) -> np.ndarray:
    d = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if w > 0 and h > 0:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return image_np

def detect_speech_bubbles(image_path: str, output_path: str = None, draw_bounding: bool = True) -> List[Dict]:
    """
    Detect speech bubbles in comic images.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the annotated image
        draw_bounding: Whether to draw bounding boxes around detected bubbles
        
    Returns:
        List of dictionaries containing the bounding box coordinates of detected bubbles
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold and morphological operations to isolate bubbles
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        area = w * h

        # Filter by size and aspect ratio to find likely speech bubbles
        if area > 1000 and 0.3 < aspect_ratio < 3:
            bubble_regions.append({"x": x, "y": y, "w": w, "h": h})
            if draw_bounding:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)

    return bubble_regions

def overwrite_speech_bubbles(image_path: str, output_path: str, bubble_regions: List[Dict] = None) -> None:
    """
    Overwrite detected speech bubbles with white polygons.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the modified image
        bubble_regions: List of bubble regions (if None, will detect them)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # If no regions provided, detect them
    if bubble_regions is None:
        bubble_regions = detect_speech_bubbles(image_path, draw_bounding=False)
    
    # Fill each bubble region with white
    for region in bubble_regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # -1 fills the rectangle
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    
    return len(bubble_regions)

def build_embedding_index(tags, model):
    embeddings = model.encode(tags, convert_to_tensor=True)
    return tags, embeddings

def search_similar_tags(query, tag_list, model, top_k=5):
    tags, embeddings = build_embedding_index(tag_list, model)
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(tags[i], scores[i]) for i in top_indices]

def generate_tag_heatmap(metadata_path="image_metadata.json", output="cooccurrence_heatmap.png"):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = list({tag for entry in data.values() for tag in entry.get("predicted_tags", {})})
    matrix = pd.DataFrame(0, index=tags, columns=tags)
    for meta in data.values():
        tag_set = [t for t, c in meta.get("predicted_tags", {}).items() if c > 0.3]
        for i in tag_set:
            for j in tag_set:
                matrix.loc[i, j] += 1
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap="coolwarm")
    plt.title("Tag Co-occurrence Heatmap")
    plt.tight_layout()
    plt.savefig(output)
    print(f"[✓] Heatmap saved to {output}")

def mask_speech_bubbles(image_path, bubbles):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in bubbles:
        draw.rectangle([x, y, x + w, y + h], fill="white")
    return img

def launch_annotation_ui(metadata_path="image_metadata.json", output_path="human_annotations.json"):
    st.title("🖍️ Human Tag Annotator")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    img_list = list(metadata.keys())
    selected = st.selectbox("Select Image", img_list)
    img = Image.open(selected)
    st.image(img, caption=selected, use_column_width=True)

    tags = list(metadata[selected].get("predicted_tags", {}).keys())
    edited = st.multiselect("Edit Tags", options=tags, default=tags)
    new_tags = st.text_input("Add new tags (comma-separated)")

    if st.button("Save Annotation"):
        annotations = {selected: list(set(edited + new_tags.split(",")))}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = {}
        existing.update(annotations)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        st.success("Annotation saved.")

def run_normalizer(args):
    normalize_metadata_file(args.input, args.output, args.tag_key, args.threshold)

def run_heatmap(args):
    generate_tag_heatmap(args.input, args.output)

def run_search(args):
    with open(args.input, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    tags = list({t for m in metadata.values() for t in m.get("predicted_tags", {})})
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = search_similar_tags(args.query, tags, model)
    print("\nTop similar tags:")
    for tag, score in results:
        print(f"{tag}: {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Image Tagging System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Normalizer
    p = subparsers.add_parser("normalize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tag_key", default="predicted_tags")
    p.add_argument("--threshold", type=float, default=0.2)
    p.set_defaults(func=run_normalizer)

    # Heatmap
    p = subparsers.add_parser("heatmap")
    p.add_argument("--input", default="image_metadata.json")
    p.add_argument("--output", default="cooccurrence_heatmap.png")
    p.set_defaults(func=run_heatmap)

    # Semantic Search
    p = subparsers.add_parser("search")
    p.add_argument("--input", required=True)
    p.add_argument("--query", required=True)
    p.set_defaults(func=run_search)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def extract_image_metadata(image_path: str) -> dict:
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            meta = {
                "resolution": f"{img.width}x{img.height}",
                "format": img.format,
                "mode": img.mode,
                "prompt": None,
                "seed": None,
                "sampler": None,
                "source_tags": {},
            }

            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (int, float, str)):
                        meta[tag] = value
                        if tag == "Prompt":
                            meta["prompt"] = value
                        elif tag == "Seed":
                            try:
                                meta["seed"] = int(value)
                            except ValueError:
                                pass
                        elif tag == "Sampler":
                            meta["sampler"] = value

            return meta
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        return {}

def find_images(base_dir: str) -> list:
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    return [p for p in Path(base_dir).rglob("*") if p.suffix.lower() in image_extensions]

def extract_image_metadata(image_path: str) -> dict:
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            meta = {
                "resolution": f"{img.width}x{img.height}",
                "format": img.format,
                "mode": img.mode,
                "prompt": None,
                "seed": None,
                "sampler": None,
                "source_tags": {},
            }

            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (int, float, str)):
                        meta[tag] = value
                        if tag == "Prompt":
                            meta["prompt"] = value
                        elif tag == "Seed":
                            try:
                                meta["seed"] = int(value)
                            except ValueError:
                                pass
                        elif tag == "Sampler":
                            meta["sampler"] = value

            return meta
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        return {}

def find_images(base_dir: str) -> list:
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    return [p for p in Path(base_dir).rglob("*") if p.suffix.lower() in image_extensions]

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

def detect_speech_bubbles(image_path: str, output_path: str = None, draw_bounding: bool = True) -> List[Dict]:
    """
    Detect speech bubbles in comic images.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the annotated image
        draw_bounding: Whether to draw bounding boxes around detected bubbles
        
    Returns:
        List of dictionaries containing the bounding box coordinates of detected bubbles
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold and morphological operations to isolate bubbles
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        area = w * h

        # Filter by size and aspect ratio to find likely speech bubbles
        if area > 1000 and 0.3 < aspect_ratio < 3:
            bubble_regions.append({"x": x, "y": y, "w": w, "h": h})
            if draw_bounding:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)

    return bubble_regions

def overwrite_speech_bubbles(image_path: str, output_path: str, bubble_regions: List[Dict] = None) -> None:
    """
    Overwrite detected speech bubbles with white polygons.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the modified image
        bubble_regions: List of bubble regions (if None, will detect them)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # If no regions provided, detect them
    if bubble_regions is None:
        bubble_regions = detect_speech_bubbles(image_path, draw_bounding=False)
    
    # Fill each bubble region with white
    for region in bubble_regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # -1 fills the rectangle
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    
    return len(bubble_regions)

def detect_speech_bubbles(image_path: str, output_path: str = None, draw_bounding: bool = True) -> List[Dict]:
    """
    Detect speech bubbles in comic images.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the annotated image
        draw_bounding: Whether to draw bounding boxes around detected bubbles
        
    Returns:
        List of dictionaries containing the bounding box coordinates of detected bubbles
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold and morphological operations to isolate bubbles
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        area = w * h

        # Filter by size and aspect ratio to find likely speech bubbles
        if area > 1000 and 0.3 < aspect_ratio < 3:
            bubble_regions.append({"x": x, "y": y, "w": w, "h": h})
            if draw_bounding:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)

    return bubble_regions

def overwrite_speech_bubbles(image_path: str, output_path: str, bubble_regions: List[Dict] = None) -> None:
    """
    Overwrite detected speech bubbles with white polygons.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the modified image
        bubble_regions: List of bubble regions (if None, will detect them)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # If no regions provided, detect them
    if bubble_regions is None:
        bubble_regions = detect_speech_bubbles(image_path, draw_bounding=False)
    
    # Fill each bubble region with white
    for region in bubble_regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # -1 fills the rectangle
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    
    return len(bubble_regions)

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

def load_metadata(data_path="image_metadata.json"):
    if not os.path.exists(data_path):
        return {}
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_tag(data, key="predicted_tags", min_conf=0.3):
    tag_groups = {}
    for image, info in data.items():
        tags = info.get(key, {})
        for tag, conf in tags.items():
            if conf >= min_conf:
                tag_groups.setdefault(tag, []).append(image)
    return tag_groups

def image_thumbnail(path, size=(150, 150)):
    try:
        img = Image.open(path)
        img.thumbnail(size)
        return img
    except:
        return None

def display_images(image_list, data, tag_filter=None, tag_type="predicted_tags", confidence_threshold=0.3):
    cols = st.columns(4)
    for idx, img_path in enumerate(image_list):
        with cols[idx % 4]:
            img = image_thumbnail(img_path)
            if img:
                st.image(img, caption=os.path.basename(img_path), use_column_width=True)
                tags = data[img_path].get(tag_type, {})
                filtered_tags = {k: v for k, v in tags.items() if v >= confidence_threshold}
                sorted_tags = sorted(filtered_tags.items(), key=lambda x: -x[1])
                for tag, conf in sorted_tags:
                    color = f"rgba(255,0,0,{conf:.2f})" if conf > 0.6 else f"rgba(0,0,255,{conf:.2f})"
                    st.markdown(f"<span style='color:{color}'>{tag}: {conf:.2f}</span>", unsafe_allow_html=True)

def find_images(base_dir):
    return [p for p in Path(base_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

def load_metadata():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def scan_and_update_images():
    images = find_images(".")
    metadata = load_metadata()
    updated = False

    for img_path in images:
        str_path = str(img_path.resolve())
        if str_path not in metadata:
            metadata[str_path] = {
                "file_name": img_path.name,
                "folder": str(img_path.parent),
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
            updated = True

    if updated:
        save_metadata(metadata)
    return metadata

def tag_images_stub(metadata):
    # Dummy implementation – this function would call the local/model API
    for path in metadata:
        if not metadata[path]["predicted_tags"]:
            metadata[path]["predicted_tags"] = {
                "red_hair": 0.87,
                "smile": 0.91,
                "school_uniform": 0.79,
            }
    save_metadata(metadata)

def build_dashboard(metadata):
    st.title("🖼️ Tag Dashboard Tool")
    st.markdown("Browse and explore image tags")

    selected_tag = st.sidebar.text_input("Filter by Tag")
    for path, info in metadata.items():
        if selected_tag and selected_tag not in info.get("predicted_tags", {}):
            continue
        st.image(path, width=256)
        st.text(f"File: {info['file_name']}")
        st.text(f"Folder: {info['folder']}")
        st.text(f"Tags: {', '.join(info.get('predicted_tags', {}).keys())}")
        
        # Your tag comparison snippet here:
        comparison = compare_tags(
            generated=info.get("source_tags", {}).get("positive", []),
            predicted=info.get("predicted_tags", {}),
            human=info.get("human_tags", [])
        )

        with st.expander("🔍 Tag Comparison"):
            st.json(comparison)
        
        st.markdown("---")

def main():
    st.sidebar.title("Options")
    if st.sidebar.button("🔄 Scan Images"):
        metadata = scan_and_update_images()
        st.sidebar.success("Images scanned.")
    else:
        metadata = load_metadata()

    if st.sidebar.button("🏷️ Run Tagger (stub)"):
        tag_images_stub(metadata)
        st.sidebar.success("Tags updated (stub).")

    metadata = load_metadata()
    build_dashboard(metadata)

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
                if norm in new_tags:
                    new_tags[norm] = max(new_tags[norm], score)
                else:
                    new_tags[norm] = score
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

# Other Code
# ------------------------------

st.set_page_config(layout="wide")

st.title("🧠 Tag & Image Metadata Dashboard")

if not metadata:
    st.warning("No metadata file found.")
else:
    # Summary Stats
    st.sidebar.header("📊 Filters & Controls")
    tag_type = st.sidebar.selectbox("Tag Source", ["predicted_tags", "source_tags.positive"])
    confidence_threshold = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.35, 0.01)

    # Group by tag
    grouped = group_by_tag(metadata, key="predicted_tags", min_conf=confidence_threshold)
    tag_counts = {tag: len(paths) for tag, paths in grouped.items()}
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    top_tags = [t[0] for t in sorted_tags[:30]]

    selected_tag = st.sidebar.selectbox("📌 Filter by Tag", ["All"] + top_tags)

    # Display histogram
    if st.sidebar.checkbox("📈 Show Tag Frequency Histogram"):
        st.subheader("📊 Tag Frequencies")
        df = pd.DataFrame(sorted_tags, columns=["Tag", "Count"])
        st.bar_chart(df.set_index("Tag"))

    # Image grid
    st.subheader("🖼️ Image Gallery")
    if selected_tag == "All":
        display_images(list(metadata.keys()), metadata, confidence_threshold=confidence_threshold)
    else:
        display_images(grouped[selected_tag], metadata, confidence_threshold=confidence_threshold)

"""
Tag Dashboard Tool – Modular Image Metadata and Tagging System

This script scans image files, extracts or updates metadata,
runs an image tagger (Danbooru-style), and provides a Streamlit dashboard for visual exploration.
"""

# Main Execution
# ------------------------------

if __name__ == "__main__":
    # Combined main blocks from all files

    # From image_tagging_cli.py:

        main()

    # From tag_dashboard_tool.py:

        main()
