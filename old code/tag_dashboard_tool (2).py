
# tag_dashboard_tool.py

import streamlit as st
import os
import json
from PIL import Image
from collections import Counter
import pandas as pd
import base64

# ---------------------
# Load JSON data
# ---------------------
def load_metadata(data_path="image_metadata.json"):
    if not os.path.exists(data_path):
        return {}
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------
# Grouping Utility
# ---------------------
def group_by_tag(data, key="predicted_tags", min_conf=0.3):
    tag_groups = {}
    for image, info in data.items():
        tags = info.get(key, {})
        for tag, conf in tags.items():
            if conf >= min_conf:
                tag_groups.setdefault(tag, []).append(image)
    return tag_groups

# ---------------------
# Render Thumbnail
# ---------------------
def image_thumbnail(path, size=(150, 150)):
    try:
        img = Image.open(path)
        img.thumbnail(size)
        return img
    except:
        return None

# ---------------------
# Display Images
# ---------------------
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

# ---------------------
# Main Streamlit App
# ---------------------
st.set_page_config(layout="wide")
st.title("🧠 Tag & Image Metadata Dashboard")

# Load data
metadata = load_metadata()

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
