#!/usr/bin/env python3
"""
Image Metadata Dashboard - Streamlit Interface

This module provides a web-based dashboard for exploring image metadata and tags.
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image
from typing import Dict, List, Optional

# Import backend functions
from backend import (
    load_metadata, 
    group_by_tag, 
    scan_and_update_images,
    update_metadata_with_categories
)

def image_thumbnail(path, size=(150, 150)):
    """Create thumbnail of image."""
    try:
        img = Image.open(path)
        img.thumbnail(size)
        return img
    except:
        return None

def display_images(image_list, data, tag_filter=None, tag_type="predicted_tags", confidence_threshold=0.3):
    """Display images in Streamlit grid with tag information."""
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

def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="Image Metadata Dashboard")
    st.title("🧠 Tag & Image Metadata Dashboard")
    
    # Sidebar for actions
    st.sidebar.header("🔧 Actions")
    
    # Scan for new images
    if st.sidebar.button("🔍 Scan for Images"):
        with st.spinner("Scanning for images..."):
            metadata = scan_and_update_images(".")
            st.sidebar.success(f"Found {len(metadata)} images")
            st.experimental_rerun()
    
    # Categorize tags
    if st.sidebar.button("🏷️ Categorize Tags"):
        with st.spinner("Categorizing tags..."):
            stats = update_metadata_with_categories()
            st.sidebar.success(f"Categorization complete: {stats}")
    
    # Load metadata
    metadata = load_metadata()
    
    if not metadata:
        st.warning("No metadata file found. Click 'Scan for Images' to generate metadata.")
        return
    
    # Display statistics
    st.sidebar.header("📊 Statistics")
    total_images = len(metadata)
    existing_images = sum(1 for meta in metadata.values() if meta.get("exists", True))
    st.sidebar.metric("Total Images", total_images)
    st.sidebar.metric("Existing Images", existing_images)
    st.sidebar.metric("Missing Images", total_images - existing_images)
    
    # Filters and controls
    st.sidebar.header("🎛️ Filters & Controls")
    tag_type = st.sidebar.selectbox("Tag Source", ["predicted_tags", "source_tags"])
    confidence_threshold = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.35, 0.01)
    
    # Group by tag
    grouped = group_by_tag(metadata, key=tag_type, min_conf=confidence_threshold)
    tag_counts = {tag: len(paths) for tag, paths in grouped.items()}
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    top_tags = [t[0] for t in sorted_tags[:30]]
    
    selected_tag = st.sidebar.selectbox("📌 Filter by Tag", ["All"] + top_tags)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖼️ Image Gallery")
        
        # Display images
        if selected_tag == "All":
            image_subset = list(metadata.keys())[:20]  # Limit for performance
        else:
            image_subset = grouped.get(selected_tag, [])[:20]
        
        if image_subset:
            display_images(image_subset, metadata, confidence_threshold=confidence_threshold)
        else:
            st.info("No images found with the selected filters.")
    
    with col2:
        st.subheader("📊 Tag Statistics")
        
        # Display tag frequency chart
        if sorted_tags:
            chart_data = pd.DataFrame(sorted_tags[:20], columns=["Tag", "Count"])
            st.bar_chart(chart_data.set_index("Tag"))
        
        # Tag details for selected tag
        if selected_tag != "All" and selected_tag in grouped:
            st.subheader(f"📌 '{selected_tag}' Details")
            st.write(f"**Images with this tag:** {len(grouped[selected_tag])}")
            
            # Show sample images with this tag
            sample_images = grouped[selected_tag][:5]
            for img_path in sample_images:
                img = image_thumbnail(img_path, size=(100, 100))
                if img:
                    st.image(img, caption=os.path.basename(img_path), width=100)
    
    # Optional: Display raw metadata for debugging
    if st.sidebar.checkbox("🔍 Show Raw Metadata (Debug)"):
        st.subheader("🔍 Raw Metadata")
        if selected_tag == "All":
            sample_path = list(metadata.keys())[0] if metadata else None
        else:
            sample_path = grouped.get(selected_tag, [None])[0]
        
        if sample_path and sample_path in metadata:
            st.json(metadata[sample_path])

# Run the dashboard
if __name__ == "__main__":
    run_dashboard()