#!/usr/bin/env python3
"""
Script to extract ALL metadata from images and save to JSON file
"""

from PIL import Image
from PIL.ExifTags import TAGS
import json
import os
import sys
from datetime import datetime

def extract_complete_metadata(image_path):
    """Extract ALL possible metadata from an image"""
    metadata = {
        "file_info": {},
        "png_text_chunks": {},
        "exif_data": {},
        "source_detection": "unknown",
        "errors": []
    }
    
    try:
        # Basic file info
        metadata["file_info"] = {
            "filename": os.path.basename(image_path),
            "full_path": image_path,
            "file_size_bytes": os.path.getsize(image_path),
            "file_extension": os.path.splitext(image_path)[1].lower()
        }
        
        with Image.open(image_path) as img:
            # Image properties
            metadata["file_info"].update({
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "resolution": f"{img.width}x{img.height}"
            })
            
            # PNG text chunks (where AI metadata is usually stored)
            if hasattr(img, 'text') and img.text:
                for key, value in img.text.items():
                    metadata["png_text_chunks"][key] = {
                        "content": value,
                        "length": len(value),
                        "is_json": False,
                        "json_data": None
                    }
                    
                    # Try to parse as JSON
                    try:
                        parsed_json = json.loads(value)
                        metadata["png_text_chunks"][key]["is_json"] = True
                        metadata["png_text_chunks"][key]["json_data"] = parsed_json
                    except json.JSONDecodeError:
                        pass
            
            # EXIF data
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                    # Convert value to string to ensure JSON serialization
                    metadata["exif_data"][tag_name] = str(value)
            
            # Source detection
            metadata["source_detection"] = detect_source_enhanced(metadata)
                
    except Exception as e:
        metadata["errors"].append(f"Error processing {image_path}: {str(e)}")
    
    return metadata

def detect_source_enhanced(metadata):
    """Enhanced source detection based on all available metadata"""
    png_chunks = metadata.get("png_text_chunks", {})
    exif_data = metadata.get("exif_data", {})
    
    # Check PNG text chunks for AI generation markers
    if "workflow" in png_chunks or "prompt" in png_chunks:
        if png_chunks.get("prompt", {}).get("is_json", False):
            return "ComfyUI"
    
    if "generation_data" in png_chunks:
        return "ComfyUI_with_generation_data"
    
    if "parameters" in png_chunks:
        params_content = png_chunks["parameters"]["content"]
        if "Steps:" in params_content and "Sampler:" in params_content:
            return "Automatic1111"
    
    # Check for other AI platforms
    for chunk_name in png_chunks:
        if any(platform in chunk_name.lower() for platform in ["tensor", "midjourney", "dalle"]):
            return f"AI_platform_{chunk_name}"
    
    # Check EXIF for editing software
    software = exif_data.get("Software", "")
    if "adobe" in software.lower():
        return "Adobe_edited"
    elif "photoshop" in software.lower():
        return "Photoshop_edited"
    
    # Check file format patterns
    file_format = metadata.get("file_info", {}).get("format", "")
    if file_format == "WEBP":
        return "Web_optimized"
    elif file_format in ["JPEG", "JPG"] and not exif_data:
        return "Web_download_stripped"
    
    return "Unknown_source"

def scan_all_images():
    """Scan all images in the examples folder"""
    base_dir = "image_examples"
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return []
    
    image_files = []
    for file in os.listdir(base_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')):
            image_files.append(os.path.join(base_dir, file))
    
    return sorted(image_files)

def main():
    print("🔍 COMPLETE METADATA EXTRACTOR")
    print("This will extract ALL metadata from your images and save to JSON")
    
    # Get all image files
    all_images = scan_all_images()
    
    if not all_images:
        print("❌ No image files found in image_examples directory")
        return
    
    print(f"\n📁 Found {len(all_images)} image files")
    print("🔄 Extracting metadata from all files...")
    
    # Extract metadata from all files
    all_metadata = {
        "extraction_info": {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(all_images),
            "script_version": "v1.0"
        },
        "files": {}
    }
    
    processed = 0
    for image_path in all_images:
        filename = os.path.basename(image_path)
        print(f"  📷 Processing: {filename} ({processed + 1}/{len(all_images)})")
        
        metadata = extract_complete_metadata(image_path)
        all_metadata["files"][filename] = metadata
        processed += 1
    
    # Save to JSON file
    output_file = "complete_metadata_extraction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Extraction complete!")
    print(f"📄 Metadata saved to: {output_file}")
    print(f"📊 File size: {os.path.getsize(output_file):,} bytes")
    
    # Quick summary
    source_counts = {}
    format_counts = {}
    files_with_metadata = 0
    
    for filename, metadata in all_metadata["files"].items():
        source = metadata["source_detection"]
        file_format = metadata["file_info"]["format"]
        
        source_counts[source] = source_counts.get(source, 0) + 1
        format_counts[file_format] = format_counts.get(file_format, 0) + 1
        
        if metadata["png_text_chunks"] or metadata["exif_data"]:
            files_with_metadata += 1
    
    print(f"\n📈 SUMMARY:")
    print(f"  Files with metadata: {files_with_metadata}/{len(all_images)}")
    print(f"  Source breakdown: {dict(source_counts)}")
    print(f"  Format breakdown: {dict(format_counts)}")
    
    print(f"\n💡 You can now share '{output_file}' for analysis!")

if __name__ == "__main__":
    main()