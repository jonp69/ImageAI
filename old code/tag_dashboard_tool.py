
"""
Tag Dashboard Tool – Modular Image Metadata and Tagging System

This script scans image files, extracts or updates metadata,
runs an image tagger (Danbooru-style), and provides a Streamlit dashboard for visual exploration.
"""

import os
import json
from pathlib import Path
from PIL import Image
import streamlit as st

from tag_comparator import compare_tags

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATA_FILE = "image_metadata.json"
TOOL_VERSION = "v0.1.0"

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

if __name__ == "__main__":
    main()
