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
import hashlib
import uuid

class UnparsedMetadataScanner:
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'scanned_files': 0,
            'skipped_small_files': 0,
            'files_with_metadata': 0,
            'unparsed_files': 0,
            'ai_generated': 0,
            'errors': 0,
            'source_breakdown': defaultdict(int),
            'unparsed_chunks': defaultdict(list),
            'large_chunks': [],
            'suspicious_files': [],
            'parsing_times': []  # New: store parsing times
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
    
    def generate_unique_filename(self, base_name: str = "unparsed_metadata_report", 
                               extension: str = ".json") -> str:
        """Generate unique filename based on timestamp and hash"""
        # Get current timestamp with microseconds
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        microseconds = now.microsecond
        
        # Create unique string from timestamp + random UUID
        unique_string = f"{timestamp}_{microseconds}_{uuid.uuid4().hex[:8]}"
        
        # Generate short hash of the unique string
        hash_obj = hashlib.sha256(unique_string.encode())
        short_hash = hash_obj.hexdigest()[:12]
        
        # Combine all parts
        unique_filename = f"{base_name}_{timestamp}_{short_hash}{extension}"
        
        # Ensure the file doesn't already exist (extremely unlikely but safe)
        counter = 1
        original_filename = unique_filename
        while os.path.exists(unique_filename):
            name_part = original_filename.replace(extension, "")
            unique_filename = f"{name_part}_v{counter}{extension}"
            counter += 1
        
        return unique_filename
    
    def is_file_large_enough(self, file_path: str, min_size_kb: int = 50) -> bool:
        """Check if file is large enough to potentially contain metadata"""
        try:
            file_size = os.path.getsize(file_path)
            min_size_bytes = min_size_kb * 1024
            return file_size >= min_size_bytes
        except OSError:
            return False  # Skip files we can't access
    
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
        """Fast scan of single image for unparsed metadata with timing"""
        start_time = time.perf_counter()  # High precision timing
        
        result = {
            'path': str(image_path),
            'filename': os.path.basename(image_path),
            'has_unparsed': False,
            'ai_likelihood': 0,
            'unparsed_chunks': [],
            'file_size': 0,
            'format': 'unknown',
            'error': None,
            'parse_time_ms': 0,  # New: parsing time
            'performance_info': {}  # New: detailed performance breakdown
        }
        
        timing_breakdown = {
            'file_access': 0,
            'image_open': 0,
            'png_chunks': 0,
            'exif_data': 0,
            'metadata_analysis': 0
        }
        
        try:
            # Time file access
            access_start = time.perf_counter()
            result['file_size'] = os.path.getsize(image_path)
            timing_breakdown['file_access'] = (time.perf_counter() - access_start) * 1000
            
            # Time image opening
            open_start = time.perf_counter()
            with Image.open(image_path) as img:
                timing_breakdown['image_open'] = (time.perf_counter() - open_start) * 1000
                result['format'] = img.format
                
                # Time PNG text chunk processing
                png_start = time.perf_counter()
                if hasattr(img, 'text') and img.text:
                    for key, value in img.text.items():
                        analysis_start = time.perf_counter()
                        analysis = self.is_unparsed_metadata(key, value)
                        analysis_time = (time.perf_counter() - analysis_start) * 1000
                        
                        timing_breakdown['metadata_analysis'] += analysis_time
                        
                        if analysis['is_unparsed']:
                            result['has_unparsed'] = True
                            result['ai_likelihood'] += analysis['ai_likelihood']
                            result['unparsed_chunks'].append({
                                'key': key,
                                'length': len(value),
                                'reasons': analysis['reasons'],
                                'chunk_type': analysis['chunk_type'],
                                'content_preview': value[:200] + '...' if len(value) > 200 else value,
                                'analysis_time_ms': analysis_time  # New: per-chunk timing
                            })
                
                timing_breakdown['png_chunks'] = (time.perf_counter() - png_start) * 1000
                
                # Time EXIF processing
                exif_start = time.perf_counter()
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, f"Tag_{tag_id}")
                        value_str = str(value)
                        
                        if len(value_str) > 100:  # Large EXIF fields
                            analysis_start = time.perf_counter()
                            analysis = self.is_unparsed_metadata(tag_name, value_str)
                            analysis_time = (time.perf_counter() - analysis_start) * 1000
                            
                            timing_breakdown['metadata_analysis'] += analysis_time
                            
                            if analysis['is_unparsed']:
                                result['has_unparsed'] = True
                                result['ai_likelihood'] += analysis['ai_likelihood']
                                result['unparsed_chunks'].append({
                                    'key': f'EXIF_{tag_name}',
                                    'length': len(value_str),
                                    'reasons': analysis['reasons'],
                                    'chunk_type': 'exif',
                                    'content_preview': value_str[:200] + '...' if len(value_str) > 200 else value_str,
                                    'analysis_time_ms': analysis_time
                                })
                
                timing_breakdown['exif_data'] = (time.perf_counter() - exif_start) * 1000
                
        except Exception as e:
            result['error'] = str(e)
            self.stats['errors'] += 1
        
        # Calculate total time
        total_time = time.perf_counter() - start_time
        result['parse_time_ms'] = total_time * 1000
        result['performance_info'] = timing_breakdown
        
        # Store timing info for statistics
        timing_record = {
            'filename': result['filename'],
            'parse_time_ms': result['parse_time_ms'],
            'file_size': result['file_size'],
            'format': result['format'],
            'chunks_found': len(result['unparsed_chunks']),
            'has_error': result['error'] is not None,
            'performance_breakdown': timing_breakdown
        }
        self.stats['parsing_times'].append(timing_record)
        
        return result
    
    def scan_directory(self, directory: str, recursive: bool = True, 
                      extensions: list = None, max_files: int = None,
                      min_size_kb: int = 50) -> list:
        """Scan directory for images with unparsed metadata"""
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']
        
        print(f"🔍 Scanning directory: {directory}")
        print(f"📁 Recursive: {recursive}")
        print(f"📄 Extensions: {extensions}")
        print(f"📏 Min file size: {min_size_kb}KB ({min_size_kb * 1024:,} bytes)")
        
        # Collect all image files
        image_files = []
        skipped_small = 0
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        
                        # Check file size
                        if self.is_file_large_enough(file_path, min_size_kb):
                            image_files.append(file_path)
                            if max_files and len(image_files) >= max_files:
                                break
                        else:
                            skipped_small += 1
                            
                if max_files and len(image_files) >= max_files:
                    break
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
                    # Check file size
                    if self.is_file_large_enough(file_path, min_size_kb):
                        image_files.append(file_path)
                        if max_files and len(image_files) >= max_files:
                            break
                    else:
                        skipped_small += 1
        
        self.stats['total_files'] = len(image_files) + skipped_small
        self.stats['skipped_small_files'] = skipped_small
        
        print(f"📊 Found {len(image_files) + skipped_small:,} total image files")
        print(f"📏 Skipped {skipped_small:,} files under {min_size_kb}KB")
        print(f"🔍 Scanning {len(image_files):,} files")
        
        # Scan files
        unparsed_files = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(image_files) - i) / rate if rate > 0 else 0
                
                # Show average parsing time
                if self.stats['parsing_times']:
                    avg_parse_time = sum(t['parse_time_ms'] for t in self.stats['parsing_times']) / len(self.stats['parsing_times'])
                    print(f"⏳ Progress: {i}/{len(image_files)} ({rate:.1f} files/sec, avg: {avg_parse_time:.1f}ms/file, ETA: {eta:.0f}s)")
                else:
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
    
    def generate_performance_report(self):
        """Generate performance troubleshooting report"""
        if not self.stats['parsing_times']:
            return
        
        # Sort by parsing time (slowest first)
        sorted_times = sorted(self.stats['parsing_times'], 
                            key=lambda x: x['parse_time_ms'], reverse=True)
        
        # Calculate statistics
        all_times = [t['parse_time_ms'] for t in self.stats['parsing_times']]
        avg_time = sum(all_times) / len(all_times)
        median_time = sorted(all_times)[len(all_times) // 2]
        max_time = max(all_times)
        min_time = min(all_times)
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print(f"📊 Parsing Time Statistics:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Median:  {median_time:.2f}ms")
        print(f"  Fastest: {min_time:.2f}ms")
        print(f"  Slowest: {max_time:.2f}ms")
        
        # Performance by format
        format_times = defaultdict(list)
        for timing in self.stats['parsing_times']:
            format_times[timing['format']].append(timing['parse_time_ms'])
        
        print(f"\n📁 Average Time by Format:")
        for fmt, times in format_times.items():
            avg_fmt_time = sum(times) / len(times)
            print(f"  {fmt}: {avg_fmt_time:.2f}ms ({len(times)} files)")
        
        # Show 10 slowest files with troubleshooting info
        print(f"\n🐌 TOP 10 SLOWEST FILES (Troubleshooting):")
        print("=" * 80)
        
        for i, timing in enumerate(sorted_times[:10]):
            print(f"\n{i+1}. {timing['filename']}")
            print(f"   ⏱️  Total time: {timing['parse_time_ms']:.2f}ms")
            print(f"   📏 File size: {timing['file_size']/1024:.1f}KB")
            print(f"   📄 Format: {timing['format']}")
            print(f"   🔍 Chunks found: {timing['chunks_found']}")
            print(f"   ❌ Has error: {timing['has_error']}")
            
            # Breakdown of where time was spent
            breakdown = timing['performance_breakdown']
            print(f"   📊 Time breakdown:")
            print(f"      File access: {breakdown['file_access']:.2f}ms")
            print(f"      Image open:  {breakdown['image_open']:.2f}ms")
            print(f"      PNG chunks:  {breakdown['png_chunks']:.2f}ms")
            print(f"      EXIF data:   {breakdown['exif_data']:.2f}ms")
            print(f"      Analysis:    {breakdown['metadata_analysis']:.2f}ms")
            
            # Calculate efficiency (time per KB)
            if timing['file_size'] > 0:
                efficiency = timing['parse_time_ms'] / (timing['file_size'] / 1024)
                print(f"   📈 Efficiency: {efficiency:.2f}ms/KB")
            
            # Identify potential issues
            issues = []
            if timing['parse_time_ms'] > avg_time * 3:
                issues.append("Extremely slow (3x+ average)")
            if breakdown['image_open'] > timing['parse_time_ms'] * 0.5:
                issues.append("Slow image opening (corrupt/large file?)")
            if breakdown['metadata_analysis'] > timing['parse_time_ms'] * 0.5:
                issues.append("Slow metadata analysis (complex chunks?)")
            if timing['file_size'] > 10 * 1024 * 1024:  # > 10MB
                issues.append("Very large file")
            if timing['has_error']:
                issues.append("Parse error occurred")
            
            if issues:
                print(f"   ⚠️  Potential issues: {', '.join(issues)}")
        
        # Find patterns in slow files
        slow_files = sorted_times[:20]  # Top 20 slowest
        slow_formats = defaultdict(int)
        slow_sizes = []
        
        for timing in slow_files:
            slow_formats[timing['format']] += 1
            slow_sizes.append(timing['file_size'])
        
        print(f"\n🔍 SLOW FILE PATTERNS:")
        print(f"📄 Formats in slowest 20:")
        for fmt, count in sorted(slow_formats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {fmt}: {count} files")
        
        if slow_sizes:
            avg_slow_size = sum(slow_sizes) / len(slow_sizes)
            print(f"📏 Average size of slowest 20: {avg_slow_size/1024:.1f}KB")
    
    def generate_report(self, unparsed_files: list, output_file: str = None, min_size_kb: int = 50):
        """Generate detailed report of unparsed metadata findings"""
        
        # Generate unique filename if not provided
        if output_file is None:
            output_file = self.generate_unique_filename("unparsed_metadata_report", ".json")
        else:
            # If user provided a filename but no extension, add .json
            if not output_file.endswith('.json'):
                output_file += '.json'
            
            # If file exists, make it unique
            if os.path.exists(output_file):
                base_name = output_file.replace('.json', '')
                output_file = self.generate_unique_filename(base_name, ".json")
                print(f"ℹ️  File exists, using unique name: {output_file}")
        
        # Sort by AI likelihood
        unparsed_files.sort(key=lambda x: x['ai_likelihood'], reverse=True)
        
        # Calculate performance statistics
        performance_stats = {}
        if self.stats['parsing_times']:
            all_times = [t['parse_time_ms'] for t in self.stats['parsing_times']]
            performance_stats = {
                "average_parse_time_ms": sum(all_times) / len(all_times),
                "median_parse_time_ms": sorted(all_times)[len(all_times) // 2],
                "slowest_parse_time_ms": max(all_times),
                "fastest_parse_time_ms": min(all_times),
                "total_parsing_time_ms": sum(all_times),
                "slowest_files": sorted(self.stats['parsing_times'], 
                                      key=lambda x: x['parse_time_ms'], reverse=True)[:10]
            }
        
        report = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "report_filename": output_file,
                "scan_id": hashlib.sha256(output_file.encode()).hexdigest()[:16],
                "min_file_size_kb": min_size_kb,
                "min_file_size_bytes": min_size_kb * 1024,
                "total_files_found": self.stats['total_files'],
                "skipped_small_files": self.stats['skipped_small_files'],
                "files_scanned": self.stats['scanned_files'],
                "files_with_unparsed_metadata": len(unparsed_files),
                "likely_ai_generated": self.stats['ai_generated'],
                "scan_errors": self.stats['errors']
            },
            "performance_stats": performance_stats,  # New section
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
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ Report successfully saved to: {output_file}")
        except Exception as e:
            # If save fails, generate emergency backup filename
            emergency_file = self.generate_unique_filename("emergency_report", ".json")
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"❌ Failed to save to {output_file}: {e}")
            print(f"💾 Emergency backup saved to: {emergency_file}")
            output_file = emergency_file
        
        # Print summary
        print(f"\n📊 UNPARSED METADATA SCAN RESULTS:")
        print(f"📁 Total files found: {self.stats['total_files']:,}")
        print(f"📏 Skipped (under {min_size_kb}KB): {self.stats['skipped_small_files']:,}")
        print(f"🔍 Files scanned: {self.stats['scanned_files']:,}")
        print(f"📝 Files with unparsed metadata: {len(unparsed_files):,}")
        print(f"🤖 Likely AI-generated: {self.stats['ai_generated']:,}")
        print(f"❌ Scan errors: {self.stats['errors']:,}")
        
        # File size statistics
        if unparsed_files:
            file_sizes = [f['file_size'] for f in unparsed_files]
            avg_size = sum(file_sizes) / len(file_sizes)
            min_size = min(file_sizes)
            max_size = max(file_sizes)
            
            print(f"\n📏 FILE SIZE STATS (unparsed files only):")
            print(f"  Average: {avg_size/1024:.1f}KB")
            print(f"  Smallest: {min_size/1024:.1f}KB")
            print(f"  Largest: {max_size/1024:.1f}KB")
        
        if unparsed_files:
            print(f"\n🏆 TOP UNPARSED CHUNK TYPES:")
            for chunk_name, count in report["statistics"]["top_unparsed_chunks"][:10]:
                print(f"  {chunk_name}: {count} files")
            
            print(f"\n🚨 HIGH PRIORITY FILES (AI likelihood > 50%):")
            high_priority = [f for f in unparsed_files if f['ai_likelihood'] > 50]
            for i, file_info in enumerate(high_priority[:10]):
                size_kb = file_info['file_size'] / 1024
                parse_time = file_info.get('parse_time_ms', 0)
                print(f"  {i+1}. {file_info['filename']} ({size_kb:.1f}KB, {parse_time:.1f}ms, score: {file_info['ai_likelihood']})")
                for chunk in file_info['unparsed_chunks'][:2]:
                    print(f"     - {chunk['key']}: {', '.join(chunk['reasons'])}")
        
        print(f"\n📄 Report file: {output_file}")
        print(f"📏 Report size: {os.path.getsize(output_file):,} bytes")
        print(f"🔑 Scan ID: {report['scan_info']['scan_id']}")
        
        # Generate performance troubleshooting report
        self.generate_performance_report()
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Scan large image datasets for unparsed metadata")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--max-files", "-m", type=int, help="Maximum files to scan")
    parser.add_argument("--output", "-o", help="Output report file (will be made unique if exists)")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=['.png', '.jpg', '.jpeg', '.webp'],
                       help="File extensions to scan")
    parser.add_argument("--min-size", "-s", type=int, default=50,
                       help="Minimum file size in KB (default: 50)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"❌ Directory not found: {args.directory}")
        sys.exit(1)
    
    scanner = UnparsedMetadataScanner()
    unparsed_files = scanner.scan_directory(
        args.directory, 
        recursive=args.recursive,
        extensions=args.extensions,
        max_files=args.max_files,
        min_size_kb=args.min_size
    )
    
    scanner.generate_report(unparsed_files, args.output, args.min_size)

if __name__ == "__main__":
    main()