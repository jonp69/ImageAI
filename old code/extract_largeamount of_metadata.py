#!/usr/bin/env python3
"""
Large Dataset Metadata Scanner
Finds AI-generated images and unknown metadata formats that need parsing
"""

from PIL import Image
from PIL.ExifTags import TAGS
import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class UnparsedMetadataScanner:
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'scanned_files': 0,
            'files_with_metadata': 0,
            'unparsed_files': 0,
            'ai_generated': 0,
            'errors': 0,
            'source_breakdown': defaultdict(int),
            'unparsed_chunks': defaultdict(list),
            'large_chunks': [],
            'suspicious_files': []
        }
        
        # Known AI metadata patterns
        self.known_ai_patterns = {
            'comfyui': ['workflow', 'prompt', 'generation_data'],
            'automatic1111': ['parameters'],
            'tensorart': ['tensorart', 'tensor'],
            'midjourney': ['midjourney', 'mj'],
            'dalle': ['dalle', 'openai'],
            'stablediffusion': ['stable-diffusion', 'sd'],
            'novelai': ['novelai', 'nai'],
            'invokeai': ['invokeai', 'invoke'],
            'fooocus': ['fooocus'],
            'forge': ['forge'],
            'vlad': ['vladmandic']
        }
        
        # Suspicious patterns that might indicate AI metadata
        self.suspicious_patterns = [
            'steps', 'sampler', 'cfg', 'seed', 'model', 'lora', 'embedding',
            'checkpoint', 'vae', 'scheduler', 'guidance', 'denoise',
            'clip_skip', 'eta', 'negative', 'prompt', 'width', 'height'
        ]
    
    def is_unparsed_metadata(self, chunk_key: str, chunk_content: str) -> dict:
        """Determine if metadata chunk is unparsed/unknown"""
        result = {
            'is_unparsed': False,
            'reasons': [],
            'ai_likelihood': 0,
            'chunk_type': 'unknown'
        }
        
        key_lower = chunk_key.lower()
        content_lower = chunk_content.lower()
        
        # Check if it's a known parseable format
        known_parseable = False
        for platform, patterns in self.known_ai_patterns.items():
            if any(pattern in key_lower for pattern in patterns):
                result['chunk_type'] = platform
                known_parseable = True
                break
        
        # If it's JSON but not from known platforms
        try:
            json_data = json.loads(chunk_content)
            if not known_parseable and isinstance(json_data, dict):
                result['is_unparsed'] = True
                result['reasons'].append('Unknown JSON structure')
                result['ai_likelihood'] += 30
        except json.JSONDecodeError:
            pass
        
        # Check for AI-related keywords in unknown chunks
        if not known_parseable:
            ai_keywords_found = []
            for keyword in self.suspicious_patterns:
                if keyword in content_lower:
                    ai_keywords_found.append(keyword)
                    result['ai_likelihood'] += 5
            
            if ai_keywords_found:
                result['is_unparsed'] = True
                result['reasons'].append(f'AI keywords found: {ai_keywords_found[:5]}')
        
        # Check for large text chunks (might contain hidden data)
        if len(chunk_content) > 1000 and not known_parseable:
            result['is_unparsed'] = True
            result['reasons'].append(f'Large unknown chunk ({len(chunk_content)} chars)')
            result['ai_likelihood'] += 10
        
        # Check for unusual chunk names
        unusual_names = [
            'usercomment', 'comment', 'description', 'software',
            'imagedescription', 'artist', 'copyright', 'title'
        ]
        if key_lower in unusual_names and len(chunk_content) > 100:
            result['is_unparsed'] = True
            result['reasons'].append('Suspicious content in standard field')
            result['ai_likelihood'] += 15
        
        return result
    
    def scan_image_fast(self, image_path: str) -> dict:
        """Fast scan of single image for unparsed metadata"""
        result = {
            'path': str(image_path),
            'filename': os.path.basename(image_path),
            'has_unparsed': False,
            'ai_likelihood': 0,
            'unparsed_chunks': [],
            'file_size': 0,
            'format': 'unknown',
            'error': None
        }
        
        try:
            result['file_size'] = os.path.getsize(image_path)
            
            with Image.open(image_path) as img:
                result['format'] = img.format
                
                # Check PNG text chunks
                if hasattr(img, 'text') and img.text:
                    for key, value in img.text.items():
                        analysis = self.is_unparsed_metadata(key, value)
                        
                        if analysis['is_unparsed']:
                            result['has_unparsed'] = True
                            result['ai_likelihood'] += analysis['ai_likelihood']
                            result['unparsed_chunks'].append({
                                'key': key,
                                'length': len(value),
                                'reasons': analysis['reasons'],
                                'chunk_type': analysis['chunk_type'],
                                'content_preview': value[:200] + '...' if len(value) > 200 else value
                            })
                
                # Check EXIF for suspicious content
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                        value_str = str(value)
                        
                        if len(value_str) > 100:  # Large EXIF fields
                            analysis = self.is_unparsed_metadata(tag_name, value_str)
                            if analysis['is_unparsed']:
                                result['has_unparsed'] = True
                                result['ai_likelihood'] += analysis['ai_likelihood']
                                result['unparsed_chunks'].append({
                                    'key': f'EXIF_{tag_name}',
                                    'length': len(value_str),
                                    'reasons': analysis['reasons'],
                                    'chunk_type': 'exif',
                                    'content_preview': value_str[:200] + '...' if len(value_str) > 200 else value_str
                                })
                
        except Exception as e:
            result['error'] = str(e)
            self.stats['errors'] += 1
        
        return result
    
    def scan_directory(self, directory: str, recursive: bool = True, 
                      extensions: list = None, max_files: int = None) -> list:
        """Scan directory for images with unparsed metadata"""
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']
        
        print(f"🔍 Scanning directory: {directory}")
        print(f"📁 Recursive: {recursive}")
        print(f"📄 Extensions: {extensions}")
        
        # Collect all image files
        image_files = []
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_files.append(os.path.join(root, file))
                        if max_files and len(image_files) >= max_files:
                            break
                if max_files and len(image_files) >= max_files:
                    break
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
                    image_files.append(file_path)
                    if max_files and len(image_files) >= max_files:
                        break
        
        self.stats['total_files'] = len(image_files)
        print(f"📊 Found {len(image_files)} image files")
        
        # Scan files
        unparsed_files = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(image_files) - i) / rate if rate > 0 else 0
                print(f"⏳ Progress: {i}/{len(image_files)} ({rate:.1f} files/sec, ETA: {eta:.0f}s)")
            
            result = self.scan_image_fast(image_path)
            self.stats['scanned_files'] += 1
            
            if result['has_unparsed']:
                unparsed_files.append(result)
                self.stats['unparsed_files'] += 1
                
                # Track high-likelihood AI files
                if result['ai_likelihood'] > 30:
                    self.stats['ai_generated'] += 1
                
                # Collect chunk statistics
                for chunk in result['unparsed_chunks']:
                    self.stats['unparsed_chunks'][chunk['key']].append(result['filename'])
            
            if result.get('unparsed_chunks') or result.get('error'):
                self.stats['files_with_metadata'] += 1
        
        elapsed = time.time() - start_time
        print(f"✅ Scan complete! Processed {len(image_files)} files in {elapsed:.1f}s")
        
        return unparsed_files
    
    def generate_report(self, unparsed_files: list, output_file: str = None):
        """Generate detailed report of unparsed metadata findings"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"unparsed_metadata_report_{timestamp}.json"
        
        # Sort by AI likelihood
        unparsed_files.sort(key=lambda x: x['ai_likelihood'], reverse=True)
        
        report = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "total_files_scanned": self.stats['scanned_files'],
                "files_with_unparsed_metadata": len(unparsed_files),
                "likely_ai_generated": self.stats['ai_generated'],
                "scan_errors": self.stats['errors']
            },
            "statistics": {
                "unparsed_chunk_frequency": dict(self.stats['unparsed_chunks']),
                "top_unparsed_chunks": sorted(
                    [(k, len(v)) for k, v in self.stats['unparsed_chunks'].items()],
                    key=lambda x: x[1], reverse=True
                )[:20]
            },
            "high_priority_files": [
                f for f in unparsed_files if f['ai_likelihood'] > 50
            ][:50],
            "all_unparsed_files": unparsed_files
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n📊 UNPARSED METADATA SCAN RESULTS:")
        print(f"📁 Total files scanned: {self.stats['scanned_files']:,}")
        print(f"🔍 Files with unparsed metadata: {len(unparsed_files):,}")
        print(f"🤖 Likely AI-generated: {self.stats['ai_generated']:,}")
        print(f"❌ Scan errors: {self.stats['errors']:,}")
        
        if unparsed_files:
            print(f"\n🏆 TOP UNPARSED CHUNK TYPES:")
            for chunk_name, count in report["statistics"]["top_unparsed_chunks"][:10]:
                print(f"  {chunk_name}: {count} files")
            
            print(f"\n🚨 HIGH PRIORITY FILES (AI likelihood > 50%):")
            high_priority = [f for f in unparsed_files if f['ai_likelihood'] > 50]
            for i, file_info in enumerate(high_priority[:10]):
                print(f"  {i+1}. {file_info['filename']} (score: {file_info['ai_likelihood']})")
                for chunk in file_info['unparsed_chunks'][:2]:
                    print(f"     - {chunk['key']}: {', '.join(chunk['reasons'])}")
        
        print(f"\n📄 Full report saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Scan large image datasets for unparsed metadata")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--max-files", "-m", type=int, help="Maximum files to scan")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=['.png', '.jpg', '.jpeg', '.webp'],
                       help="File extensions to scan")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        sys.exit(1)
    
    scanner = UnparsedMetadataScanner()
    unparsed_files = scanner.scan_directory(
        args.directory, 
        recursive=args.recursive,
        extensions=args.extensions,
        max_files=args.max_files
    )
    
    scanner.generate_report(unparsed_files, args.output)

if __name__ == "__main__":
    main()