#!/usr/bin/env python3
"""
Script to inspect actual metadata in your image files
"""

from PIL import Image
import json
import os
import sys

def inspect_image_detailed(image_path):
    """Detailed inspection of actual image metadata"""
    print(f"\n" + "="*60)
    print(f"🔍 ANALYZING: {os.path.basename(image_path)}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        with Image.open(image_path) as img:
            print(f"📏 Dimensions: {img.width} x {img.height}")
            print(f"📁 Format: {img.format}")
            print(f"🎨 Mode: {img.mode}")
            print(f"📊 File size: {os.path.getsize(image_path):,} bytes")
            
            # Check PNG text chunks (where ComfyUI/A1111 store data)
            if hasattr(img, 'text') and img.text:
                print(f"\n📝 PNG TEXT CHUNKS FOUND ({len(img.text)}):")
                for key, value in img.text.items():
                    print(f"  🔑 {key}:")
                    if len(value) > 200:
                        print(f"    📄 {value[:200]}...")
                        print(f"    📏 [Total length: {len(value)} characters]")
                        
                        # Try to parse as JSON
                        if key.lower() in ['workflow', 'prompt']:
                            try:
                                parsed = json.loads(value)
                                print(f"    ✅ Valid JSON structure detected")
                                if isinstance(parsed, dict):
                                    print(f"    🗂️  JSON keys: {list(parsed.keys())[:5]}...")
                            except:
                                print(f"    ❌ Not valid JSON")
                    else:
                        print(f"    📄 {value}")
            else:
                print("\n📝 PNG TEXT CHUNKS: None found")
            
            # Check EXIF data
            exif = img.getexif()
            if exif:
                print(f"\n📷 EXIF DATA FOUND ({len(exif)} entries):")
                for tag_id, value in list(exif.items())[:10]:  # Show first 10
                    tag_name = Image.ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"  🏷️  {tag_name}: {value_str}")
                
                if len(exif) > 10:
                    print(f"  ... and {len(exif) - 10} more EXIF entries")
            else:
                print("\n📷 EXIF DATA: None found")
                
    except Exception as e:
        print(f"❌ Error opening file: {e}")

def detect_source_from_metadata(image_path):
    """Try to detect the source based on actual metadata"""
    try:
        with Image.open(image_path) as img:
            if hasattr(img, 'text') and img.text:
                # Check for ComfyUI markers
                if 'workflow' in img.text or 'prompt' in img.text:
                    return "ComfyUI"
                
                # Check for A1111 markers
                if 'parameters' in img.text:
                    params = img.text['parameters']
                    if 'Steps:' in params and 'Sampler:' in params:
                        return "Automatic1111"
                
                # Check for other known formats
                if any(key.startswith('tensor') for key in img.text.keys()):
                    return "TensorArt"
                    
            return "Unknown/Generic"
                    
    except Exception as e:
        return f"Error: {e}"

def scan_all_images():
    """Scan all images in the examples folder"""
    base_dir = "image_examples"
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return []
    
    image_files = []
    for file in os.listdir(base_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_files.append(os.path.join(base_dir, file))
    
    return sorted(image_files)

# Updated files based on actual directory structure
def main():
    print("🔍 IMAGE METADATA INSPECTOR")
    print("This will show you exactly what metadata is in your files")
    
    # Get all image files from the directory
    all_images = scan_all_images()
    
    if not all_images:
        print("❌ No image files found in image_examples directory")
        return
    
    print(f"\n📁 Found {len(all_images)} image files")
    
    # Sample a few different types for detailed inspection
    sample_files = [
        # PNG files (likely to have metadata)
        next((f for f in all_images if f.endswith('.png')), None),
        # JPG files 
        next((f for f in all_images if f.endswith('.jpg')), None),
        # WEBP files
        next((f for f in all_images if f.endswith('.webp')), None),
        # JPEG files
        next((f for f in all_images if f.endswith('.jpeg')), None),
    ]
    
    # Remove None entries
    sample_files = [f for f in sample_files if f is not None]
    
    # Add a few more random samples
    import random
    additional_samples = random.sample(all_images, min(3, len(all_images)))
    sample_files.extend(additional_samples)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_samples = []
    for f in sample_files:
        if f not in seen:
            seen.add(f)
            unique_samples.append(f)
    
    print(f"\n🎯 INSPECTING {len(unique_samples)} SAMPLE FILES:")
    for file_path in unique_samples:
        source = detect_source_from_metadata(file_path)
        print(f"\n🎯 DETECTED SOURCE: {source}")
        inspect_image_detailed(file_path)
    
    # Quick overview of all files
    print(f"\n" + "="*60)
    print("📊 QUICK OVERVIEW OF ALL FILES")
    print("="*60)
    
    source_counts = {}
    format_counts = {}
    
    for file_path in all_images:
        source = detect_source_from_metadata(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        source_counts[source] = source_counts.get(source, 0) + 1
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    print("\n📈 SOURCE BREAKDOWN:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} files")
    
    print("\n📂 FORMAT BREAKDOWN:")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count} files")
    
    print("\n" + "="*60)
    print("✅ INSPECTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()