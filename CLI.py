#!/usr/bin/env python3
"""
Image Metadata and Tagging Tool - Main CLI Entry Point

This is the main CLI interface. For the dashboard, run: streamlit run dashboard.py
"""

import argparse
from backend import (
    scan_and_update_images,
    detect_speech_bubbles,
    update_metadata_with_categories
)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Image Metadata and Tagging Tool")
    parser.add_argument("--scan", action="store_true", help="Scan directory for images")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
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
    
    if args.dashboard:
        print("Launch Streamlit dashboard with: streamlit run dashboard.py")
        print("Or run: python -m streamlit run dashboard.py")

if __name__ == "__main__":
    main()