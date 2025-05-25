#!/usr/bin/env python3
"""
Image Metadata and Tagging System - Backend Functions (Restored)

This module contains all the core functionality including restored functions
from the old code that were missing in the consolidation.
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
import time
import shutil
from datetime import datetime
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from collections import Counter, defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from transformers import AutoProcessor, AutoModelForImageClassification
from typing import Dict, List, Set, Optional, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Global Variables
# ------------------------------
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_FILE = "image_metadata.json"
TOOL_VERSION = "v0.2.0"

# Enhanced tag synonym mapping
TAG_SYNONYM_MAP = {
    "redhead": "red_hair",
    "blonde": "blonde_hair", 
    "brunette": "brown_hair",
    "happy": "smile",
    "grin": "smile",
    "smiling": "smile",
    "school girl": "school_uniform",
    "uniform": "school_uniform"
}

# Default tag categories
DEFAULT_TAG_CATEGORIES = {
    "hair_color": ["blonde_hair", "brown_hair", "black_hair", "red_hair", "white_hair", "silver_hair"],
    "expression": ["smile", "serious", "angry", "sad", "surprised", "crying", "laughing"],
    "clothing": ["school_uniform", "dress", "shirt", "jacket", "swimsuit", "bikini"],
    "body_parts": ["long_hair", "short_hair", "blue_eyes", "brown_eyes", "green_eyes"],
    "background": ["indoors", "outdoors", "simple_background", "detailed_background"],
    "pose": ["standing", "sitting", "lying", "walking", "running"],
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
                            self.metadata[full_path]["last_seen"] = datetime.now().isoformat()
                        else:
                            self.metadata[full_path] = {
                                "exists": True,
                                "last_modified": os.path.getmtime(full_path),
                                "size": os.path.getsize(full_path),
                                "last_seen": datetime.now().isoformat(),
                                "hash": None
                            }
        
        for path in list(self.metadata.keys()):
            if path not in existing_files:
                self.metadata[path]["exists"] = False
                self.metadata[path]["deleted_date"] = datetime.now().isoformat()
        
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
            
        elif check_method == "hash":
            by_hash = {}
            for path, info in self.metadata.items():
                if info.get("exists", True):
                    if not info.get("hash"):
                        info["hash"] = hash_file(path)
                    file_hash = info["hash"]
                    by_hash.setdefault(file_hash, []).append(path)
            duplicates = {hash_val: paths for hash_val, paths in by_hash.items() if len(paths) > 1}
            
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

class DirectoryWatcher(FileSystemEventHandler):
    """Monitor directory for new images and auto-process them."""
    
    def __init__(self, callback_func, extensions=SUPPORTED_EXTENSIONS):
        self.callback_func = callback_func
        self.extensions = extensions
        
    def on_created(self, event):
        if not event.is_directory:
            if any(event.src_path.lower().endswith(ext) for ext in self.extensions):
                print(f"New image detected: {event.src_path}")
                self.callback_func(event.src_path)

# Utility Functions
# ------------------------------
def hash_file(filepath: str) -> str:
    """Generate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
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

def backup_metadata(metadata_path="image_metadata.json", backup_dir="backups"):
    """Create timestamped backup of metadata file."""
    if not os.path.exists(metadata_path):
        return None
        
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"metadata_backup_{timestamp}.json")
    shutil.copy2(metadata_path, backup_path)
    print(f"Metadata backed up to: {backup_path}")
    return backup_path

# RESTORED: Tag Normalization Functions
# ------------------------------
def normalize_tag_format(tag: str) -> str:
    """Convert tag to standard format (lowercase, underscores)."""
    return tag.lower().replace(" ", "_").replace("-", "_")

def apply_synonym_mapping(tags: List[str], synonym_map: Dict[str, str] = None) -> List[str]:
    """Replace synonymous tags with standard versions."""
    if synonym_map is None:
        synonym_map = TAG_SYNONYM_MAP
    
    normalized = []
    for tag in tags:
        normalized_tag = normalize_tag_format(tag)
        # Apply synonym mapping
        mapped_tag = synonym_map.get(normalized_tag, normalized_tag)
        normalized.append(mapped_tag)
    
    return list(set(normalized))  # Remove duplicates

def merge_duplicate_tags(tag_dict: Dict[str, float]) -> Dict[str, float]:
    """Merge tags that are duplicates after normalization."""
    merged = {}
    for tag, confidence in tag_dict.items():
        normalized = normalize_tag_format(tag)
        mapped = TAG_SYNONYM_MAP.get(normalized, normalized)
        
        if mapped in merged:
            # Keep highest confidence
            merged[mapped] = max(merged[mapped], confidence)
        else:
            merged[mapped] = confidence
    
    return merged

def validate_tag_consistency(metadata: Dict[str, dict]) -> Dict[str, List[str]]:
    """Check for tag conflicts and inconsistencies."""
    issues = defaultdict(list)
    
    for image_path, meta in metadata.items():
        predicted_tags = meta.get("predicted_tags", {})
        source_tags = meta.get("source_tags", {})
        
        # Check for conflicting tags
        conflicting_pairs = [
            ("male", "female"),
            ("1girl", "1boy"),
            ("blonde_hair", "brown_hair"),
            ("short_hair", "long_hair")
        ]
        
        all_tags = set(predicted_tags.keys()) | set(source_tags.keys())
        for tag1, tag2 in conflicting_pairs:
            if tag1 in all_tags and tag2 in all_tags:
                issues[image_path].append(f"Conflicting tags: {tag1} vs {tag2}")
    
    return dict(issues)

# RESTORED: Accuracy Evaluation Functions
# ------------------------------
def load_human_annotations(file_path: str) -> Dict[str, List[str]]:
    """Load human-annotated tags from CSV or JSON file."""
    annotations = {}
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for image_path, tags in data.items():
                annotations[image_path] = tags if isinstance(tags, list) else [tags]
    
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            image_path = row['image_path']
            tags = row['tags'].split(',') if isinstance(row['tags'], str) else []
            annotations[image_path] = [tag.strip() for tag in tags]
    
    return annotations

def batch_evaluate_accuracy(image_list: List[str], metadata: Dict[str, dict], 
                          human_annotations: Dict[str, List[str]], 
                          conf_threshold: float = 0.3) -> Dict[str, dict]:
    """Evaluate accuracy for multiple images at once."""
    results = {}
    
    for image_path in image_list:
        if image_path in metadata and image_path in human_annotations:
            predicted = metadata[image_path].get("predicted_tags", {})
            human = human_annotations[image_path]
            
            results[image_path] = compare_tag_sets(predicted, human, conf_threshold)
    
    return results

def generate_accuracy_report(results: Dict[str, dict], output_path: str = "accuracy_report.html"):
    """Generate detailed accuracy report in HTML format."""
    if not results:
        return
    
    # Calculate overall statistics
    all_precision = [r["precision"] for r in results.values()]
    all_recall = [r["recall"] for r in results.values()]
    all_f1 = [r["f1_score"] for r in results.values()]
    
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tag Accuracy Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .image-result {{ margin: 20px 0; border: 1px solid #ccc; padding: 10px; }}
            .tag-list {{ display: inline-block; margin: 5px; padding: 3px 8px; border-radius: 3px; }}
            .correct {{ background-color: #d4edda; color: #155724; }}
            .false-positive {{ background-color: #f8d7da; color: #721c24; }}
            .false-negative {{ background-color: #fff3cd; color: #856404; }}
        </style>
    </head>
    <body>
        <h1>Tag Accuracy Report</h1>
        <div class="summary">
            <h2>Overall Statistics</h2>
            <p><strong>Average Precision:</strong> {avg_precision:.3f}</p>
            <p><strong>Average Recall:</strong> {avg_recall:.3f}</p>
            <p><strong>Average F1 Score:</strong> {avg_f1:.3f}</p>
            <p><strong>Images Evaluated:</strong> {len(results)}</p>
        </div>
    """
    
    for image_path, result in results.items():
        image_name = os.path.basename(image_path)
        html_content += f"""
        <div class="image-result">
            <h3>{image_name}</h3>
            <p><strong>Precision:</strong> {result['precision']:.3f} | 
               <strong>Recall:</strong> {result['recall']:.3f} | 
               <strong>F1:</strong> {result['f1_score']:.3f}</p>
            
            <div>
                <strong>Correct Tags:</strong><br>
                {' '.join([f'<span class="tag-list correct">{tag}</span>' for tag in result['true_positives']])}
            </div>
            
            <div>
                <strong>False Positives:</strong><br>
                {' '.join([f'<span class="tag-list false-positive">{tag}</span>' for tag in result['false_positives']])}
            </div>
            
            <div>
                <strong>Missed Tags:</strong><br>
                {' '.join([f'<span class="tag-list false-negative">{tag}</span>' for tag in result['false_negatives']])}
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Accuracy report saved to: {output_path}")

def plot_confusion_matrix(results: Dict[str, dict], save_path: str = "confusion_matrix.png"):
    """Create visualization for tag accuracy metrics."""
    if not results:
        return
    
    precision_scores = [r["precision"] for r in results.values()]
    recall_scores = [r["recall"] for r in results.values()]
    f1_scores = [r["f1_score"] for r in results.values()]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Precision distribution
    axes[0, 0].hist(precision_scores, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Precision Distribution')
    axes[0, 0].set_xlabel('Precision')
    axes[0, 0].set_ylabel('Frequency')
    
    # Recall distribution
    axes[0, 1].hist(recall_scores, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Recall Distribution')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Frequency')
    
    # F1 distribution
    axes[1, 0].hist(f1_scores, bins=20, alpha=0.7, color='red')
    axes[1, 0].set_title('F1 Score Distribution')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # Precision vs Recall scatter
    axes[1, 1].scatter(precision_scores, recall_scores, alpha=0.6)
    axes[1, 1].set_title('Precision vs Recall')
    axes[1, 1].set_xlabel('Precision')
    axes[1, 1].set_ylabel('Recall')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {save_path}")

# RESTORED: Export/Import Functions
# ------------------------------
def export_tag_list(metadata: Dict[str, dict], format_type: str = "csv", 
                   output_path: str = None, tag_source: str = "predicted_tags") -> str:
    """Export tags in various formats for external use."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tag_export_{timestamp}.{format_type}"
    
    # Collect all tags
    all_tags = set()
    image_tags = {}
    
    for image_path, meta in metadata.items():
        tags = meta.get(tag_source, {})
        if isinstance(tags, dict):
            # For predicted tags with confidence
            tag_list = list(tags.keys())
        else:
            # For simple tag lists
            tag_list = tags if isinstance(tags, list) else []
        
        image_tags[image_path] = tag_list
        all_tags.update(tag_list)
    
    if format_type == "csv":
        df_data = []
        for image_path, tags in image_tags.items():
            df_data.append({
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "tags": ",".join(tags),
                "tag_count": len(tags)
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
    
    elif format_type == "json":
        export_data = {
            "export_date": datetime.now().isoformat(),
            "tag_source": tag_source,
            "total_images": len(image_tags),
            "unique_tags": len(all_tags),
            "data": image_tags
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    elif format_type == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Tag Export\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total unique tags: {len(all_tags)}\n\n")
            
            for tag in sorted(all_tags):
                f.write(f"{tag}\n")
    
    print(f"Tags exported to: {output_path}")
    return output_path

def import_tag_annotations(file_path: str, target_field: str = "human_tags") -> Dict[str, List[str]]:
    """Import human annotations from external files."""
    annotations = {}
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            image_path = row.get('image_path', '')
            tags_str = row.get('tags', '')
            
            if image_path and tags_str:
                tags = [tag.strip() for tag in tags_str.split(',')]
                annotations[image_path] = tags
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'data' in data:
                annotations = data['data']
            else:
                annotations = data
    
    print(f"Imported {len(annotations)} annotations from {file_path}")
    return annotations

def generate_report_html(metadata: Dict[str, dict], output_path: str = "collection_report.html"):
    """Generate comprehensive HTML report of the collection."""
    # Analyze the collection
    total_images = len(metadata)
    existing_images = sum(1 for meta in metadata.values() if meta.get("exists", True))
    
    # Tag statistics
    all_predicted_tags = defaultdict(int)
    all_source_tags = defaultdict(int)
    
    for meta in metadata.values():
        pred_tags = meta.get("predicted_tags", {})
        source_tags = meta.get("source_tags", {})
        
        for tag in pred_tags:
            all_predicted_tags[tag] += 1
        
        for tag in source_tags:
            all_source_tags[tag] += 1
    
    # Most common tags
    top_predicted = sorted(all_predicted_tags.items(), key=lambda x: x[1], reverse=True)[:20]
    top_source = sorted(all_source_tags.items(), key=lambda x: x[1], reverse=True)[:20]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Collection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 30px 0; }}
            .stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .tag-table {{ border-collapse: collapse; width: 100%; }}
            .tag-table th, .tag-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .tag-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Image Collection Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Collection Overview</h2>
            <div class="stats">
                <p><strong>Total Images:</strong> {total_images}</p>
                <p><strong>Existing Images:</strong> {existing_images}</p>
                <p><strong>Missing Images:</strong> {total_images - existing_images}</p>
                <p><strong>Unique Predicted Tags:</strong> {len(all_predicted_tags)}</p>
                <p><strong>Unique Source Tags:</strong> {len(all_source_tags)}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Most Common Predicted Tags</h2>
            <table class="tag-table">
                <tr><th>Tag</th><th>Frequency</th></tr>
                {''.join([f'<tr><td>{tag}</td><td>{count}</td></tr>' for tag, count in top_predicted])}
            </table>
        </div>
        
        <div class="section">
            <h2>Most Common Source Tags</h2>
            <table class="tag-table">
                <tr><th>Tag</th><th>Frequency</th></tr>
                {''.join([f'<tr><td>{tag}</td><td>{count}</td></tr>' for tag, count in top_source])}
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Collection report saved to: {output_path}")

# RESTORED: Directory Watching
# ------------------------------
def watch_directory(path: str, callback_func, extensions: Set[str] = SUPPORTED_EXTENSIONS):
    """Monitor directory for new images and auto-process them."""
    event_handler = DirectoryWatcher(callback_func, extensions)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    
    print(f"Watching directory: {path}")
    print("Press Ctrl+C to stop watching...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Stopped watching directory.")
    
    observer.join()

def process_batch_queue(operations: List[Dict], metadata_path: str = "image_metadata.json"):
    """Process a queue of batch operations."""
    metadata = load_metadata(metadata_path)
    
    for operation in operations:
        op_type = operation.get("type")
        target = operation.get("target")
        params = operation.get("params", {})
        
        if op_type == "normalize_tags":
            if target in metadata:
                predicted_tags = metadata[target].get("predicted_tags", {})
                normalized = merge_duplicate_tags(predicted_tags)
                metadata[target]["predicted_tags"] = normalized
                print(f"Normalized tags for: {target}")
        
        elif op_type == "extract_metadata":
            if os.path.exists(target):
                metadata[target] = extract_image_metadata(target)
                print(f"Extracted metadata for: {target}")
        
        elif op_type == "detect_speech_bubbles":
            if os.path.exists(target):
                bubbles = detect_speech_bubbles(target)
                metadata[target]["speech_bubbles"] = bubbles
                print(f"Detected {len(bubbles)} speech bubbles in: {target}")
    
    save_metadata(metadata, metadata_path)
    print(f"Processed {len(operations)} batch operations")

# Keep existing functions from the original backend...
# (Include all the existing functions from the previous backend.py)

# All the original functions remain the same:
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
    """Main CLI entry point with restored functionality."""
    parser = argparse.ArgumentParser(description="Image Metadata and Tagging Tool (Restored)")
    parser.add_argument("--scan", action="store_true", help="Scan directory for images")
    parser.add_argument("--detect-bubbles", help="Detect speech bubbles in image")
    parser.add_argument("--categorize", action="store_true", help="Categorize tags")
    parser.add_argument("--normalize-tags", action="store_true", help="Normalize tag formats")
    parser.add_argument("--validate-tags", action="store_true", help="Validate tag consistency")
    parser.add_argument("--export-tags", help="Export tags (csv/json/txt)")
    parser.add_argument("--import-annotations", help="Import human annotations")
    parser.add_argument("--accuracy-report", action="store_true", help="Generate accuracy report")
    parser.add_argument("--backup", action="store_true", help="Backup metadata")
    parser.add_argument("--watch", help="Watch directory for new images")
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
    
    if args.normalize_tags:
        metadata = load_metadata()
        for path, meta in metadata.items():
            if "predicted_tags" in meta:
                normalized = merge_duplicate_tags(meta["predicted_tags"])
                meta["predicted_tags"] = normalized
        save_metadata(metadata)
        print("Tag normalization complete")
    
    if args.validate_tags:
        metadata = load_metadata()
        issues = validate_tag_consistency(metadata)
        if issues:
            print(f"Found tag consistency issues in {len(issues)} images")
            for path, problems in issues.items():
                print(f"  {path}: {problems}")
        else:
            print("No tag consistency issues found")
    
    if args.export_tags:
        metadata = load_metadata()
        format_type = args.export_tags
        export_tag_list(metadata, format_type)
    
    if args.import_annotations:
        annotations = import_tag_annotations(args.import_annotations)
        metadata = load_metadata()
        for path, tags in annotations.items():
            if path in metadata:
                metadata[path]["human_tags"] = tags
        save_metadata(metadata)
        print("Annotations imported successfully")
    
    if args.accuracy_report:
        metadata = load_metadata()
        # Find images with both predicted and human tags
        evaluation_images = [path for path, meta in metadata.items() 
                           if "predicted_tags" in meta and "human_tags" in meta]
        
        if evaluation_images:
            human_annotations = {path: metadata[path]["human_tags"] for path in evaluation_images}
            results = batch_evaluate_accuracy(evaluation_images, metadata, human_annotations)
            generate_accuracy_report(results)
            plot_confusion_matrix(results)
        else:
            print("No images found with both predicted and human tags")
    
    if args.backup:
        backup_path = backup_metadata()
        print(f"Metadata backed up to: {backup_path}")
    
    if args.watch:
        def on_new_image(image_path):
            metadata = load_metadata()
            metadata[image_path] = extract_image_metadata(image_path)
            save_metadata(metadata)
            print(f"Processed new image: {image_path}")
        
        watch_directory(args.watch, on_new_image)

if __name__ == "__main__":
    main()