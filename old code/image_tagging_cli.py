# image_tagging_cli.py
# CLI wrappers for each module + Semantic Search + Visual Graph + Human Annotation UI

import argparse
import json
import os
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from PIL import Image, ImageDraw
from tag_normalizer import normalize_metadata_file
from tag_comparator import compare_tags

# --------------------
# Semantic Search
# --------------------
def build_embedding_index(tags, model):
    embeddings = model.encode(tags, convert_to_tensor=True)
    return tags, embeddings

def search_similar_tags(query, tag_list, model, top_k=5):
    tags, embeddings = build_embedding_index(tag_list, model)
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(tags[i], scores[i]) for i in top_indices]

# --------------------
# Tag Co-occurrence Graph
# --------------------
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

# --------------------
# Export Textless Speech Panels
# --------------------
def mask_speech_bubbles(image_path, bubbles):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in bubbles:
        draw.rectangle([x, y, x + w, y + h], fill="white")
    return img

# --------------------
# Human Annotation UI
# --------------------
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

# --------------------
# CLI Entrypoint
# --------------------
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

if __name__ == "__main__":
    main()
