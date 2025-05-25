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
            'parsing_times': [],  # New: store parsing times
            'directory_limits': {},  # New: track directory file limits
            'limited_directories': []  # New: directories that hit the limit
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
        min_size_kb: int = 50, max_files_per_dir: int = None) -> list:
        """Scan directory for images with unparsed metadata"""
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']
        
        print(f"🔍 Scanning directory: {directory}")
        print(f"📁 Recursive: {recursive}")
        print(f"📄 Extensions: {extensions}")
        print(f"📏 Min file size: {min_size_kb}KB ({min_size_kb * 1024:,} bytes)")
        if max_files_per_dir:
            print(f"📂 Max files per subdirectory: {max_files_per_dir:,}")
        print(f"🚀 Will ramp up to 300+/sec before monitoring for slowdowns below 200/sec")
        
        # Performance monitoring variables
        performance_window = 100  # Check every 100 files
        min_acceptable_rate = 200.0  # files per second
        ramp_up_target_rate = 300.0  # Target rate to reach before monitoring
        ramp_up_completed = False
        slow_detections = 0
        max_slow_detections = 3  # Abort after 3 consecutive slow periods
        
        # Collect all image files with directory limiting
        image_files = []
        skipped_small = 0
        directory_file_counts = defaultdict(int)  # Track files per directory
        limited_dirs = []  # Track directories that hit the limit
        
        if recursive:
            # Group files by directory first
            dir_files = defaultdict(list)
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        
                        # Check file size
                        if self.is_file_large_enough(file_path, min_size_kb):
                            dir_files[root].append(file_path)
                        else:
                            skipped_small += 1
            
            # Apply per-directory limits and collect files
            for dir_path, files_in_dir in dir_files.items():
                files_to_add = files_in_dir
                
                if max_files_per_dir and len(files_in_dir) > max_files_per_dir:
                    files_to_add = files_in_dir[:max_files_per_dir]
                    limited_dirs.append({
                        'directory': dir_path,
                        'total_files': len(files_in_dir),
                        'selected_files': max_files_per_dir,
                        'skipped_files': len(files_in_dir) - max_files_per_dir
                    })
                    print(f"📂 Limited directory: {dir_path}")
                    print(f"   📊 Found {len(files_in_dir)} files, taking first {max_files_per_dir}")
                
                image_files.extend(files_to_add)
                directory_file_counts[dir_path] = len(files_to_add)
                
                # Check global max_files limit
                if max_files and len(image_files) >= max_files:
                    image_files = image_files[:max_files]
                    break
        else:
            # Non-recursive: treat the main directory as a single unit
            files_in_main_dir = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
                    # Check file size
                    if self.is_file_large_enough(file_path, min_size_kb):
                        files_in_main_dir.append(file_path)
                    else:
                        skipped_small += 1
            
            # Apply per-directory limit to main directory
            if max_files_per_dir and len(files_in_main_dir) > max_files_per_dir:
                image_files = files_in_main_dir[:max_files_per_dir]
                limited_dirs.append({
                    'directory': directory,
                    'total_files': len(files_in_main_dir),
                    'selected_files': max_files_per_dir,
                    'skipped_files': len(files_in_main_dir) - max_files_per_dir
                })
                print(f"📂 Limited main directory: {directory}")
                print(f"   📊 Found {len(files_in_main_dir)} files, taking first {max_files_per_dir}")
            else:
                image_files = files_in_main_dir
            
            directory_file_counts[directory] = len(image_files)
            
            # Apply global max_files limit
            if max_files and len(image_files) > max_files:
                image_files = image_files[:max_files]
        
        # Store directory limiting stats
        self.stats['directory_limits'] = dict(directory_file_counts)
        self.stats['limited_directories'] = limited_dirs
        
        # Calculate total files found before limiting
        total_files_found = sum(dir_info['total_files'] for dir_info in limited_dirs) + \
                          sum(count for dir_path, count in directory_file_counts.items() 
                              if not any(ld['directory'] == dir_path for ld in limited_dirs))
        total_files_found += skipped_small
        
        self.stats['total_files'] = total_files_found
        self.stats['skipped_small_files'] = skipped_small
        
        print(f"📊 Found {total_files_found:,} total image files")
        print(f"📏 Skipped {skipped_small:,} files under {min_size_kb}KB")
        if limited_dirs:
            total_skipped_by_limit = sum(ld['skipped_files'] for ld in limited_dirs)
            print(f"📂 Limited {len(limited_dirs)} directories, skipped {total_skipped_by_limit:,} files")
        print(f"🔍 Scanning {len(image_files):,} files")
        
        # Scan files
        unparsed_files = []
        start_time = time.time()
        window_start_time = start_time
        last_100_start = start_time
        
        for i, image_path in enumerate(image_files):
            current_time = time.time()
            
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
            
            # Performance check every 100 files
            if (i + 1) % performance_window == 0:
                window_elapsed = current_time - last_100_start
                window_rate = performance_window / window_elapsed if window_elapsed > 0 else 0
                
                # Calculate overall rate
                overall_elapsed = current_time - start_time
                overall_rate = (i + 1) / overall_elapsed if overall_elapsed > 0 else 0
                eta = (len(image_files) - i - 1) / overall_rate if overall_rate > 0 else 0
                
                # Show average parsing time
                avg_parse_time = 0
                if self.stats['parsing_times']:
                    recent_times = self.stats['parsing_times'][-performance_window:]
                    avg_parse_time = sum(t['parse_time_ms'] for t in recent_times) / len(recent_times)
                
                # Check if we've reached ramp-up target
                if not ramp_up_completed:
                    if window_rate >= ramp_up_target_rate:
                        ramp_up_completed = True
                        print(f"🚀 RAMP-UP COMPLETE! Reached {window_rate:.1f}/s (target: {ramp_up_target_rate}/s)")
                        print(f"   📊 Now monitoring for slowdowns below {min_acceptable_rate}/s")
                        print(f"   ✅ Files: {i+1}/{len(image_files)} | Overall: {overall_rate:.1f}/s | Avg parse: {avg_parse_time:.1f}ms")
                    else:
                        # Still ramping up - show encouraging progress
                        ramp_emoji = "🔥" if window_rate > 250 else "🚀" if window_rate > 150 else "⏳"
                        progress_pct = (window_rate / ramp_up_target_rate) * 100
                        print(f"{ramp_emoji} RAMP-UP: {i+1}/{len(image_files)} | "
                              f"Rate: {window_rate:.1f}/s ({progress_pct:.0f}% of target) | "
                              f"Overall: {overall_rate:.1f}/s | Parse: {avg_parse_time:.1f}ms | ETA: {eta:.0f}s")
                else:
                    # Ramp-up completed - now monitor for slowdowns
                    if window_rate < min_acceptable_rate:
                        slow_detections += 1
                        print(f"🐌 SLOW WINDOW {slow_detections}/{max_slow_detections}: {i+1}/{len(image_files)} | "
                              f"Rate: {window_rate:.1f}/s (target: {min_acceptable_rate}/s) | "
                              f"Avg parse: {avg_parse_time:.1f}ms | ETA: {eta:.0f}s")
                        
                        # Show immediate diagnostics
                        if slow_detections == 1:
                            print(f"   📊 Diagnostics:")
                            print(f"   - Recent avg parse time: {avg_parse_time:.1f}ms")
                            if self.stats['parsing_times']:
                                recent_max = max(t['parse_time_ms'] for t in self.stats['parsing_times'][-performance_window:])
                                recent_slow_files = [t['filename'] for t in self.stats['parsing_times'][-performance_window:] 
                                                   if t['parse_time_ms'] > avg_parse_time * 2]
                                print(f"   - Slowest recent file: {recent_max:.1f}ms")
                                if recent_slow_files:
                                    print(f"   - Files taking 2x+ avg time: {len(recent_slow_files)}")
                        
                        # Abort after consecutive slow periods
                        if slow_detections >= max_slow_detections:
                            print(f"\n🚨 ABORTING: {slow_detections} consecutive slow periods detected!")
                            print(f"📊 Processed {i+1} files in {overall_elapsed:.1f}s before aborting")
                            print(f"🐌 Current rate: {window_rate:.1f} files/second (target: {min_acceptable_rate})")
                            print(f"✅ Ramp-up was completed successfully before slowdown")
                            
                            # Store abort reason in stats
                            self.stats['abort_reason'] = f"Processing rate dropped to {window_rate:.1f} files/sec after ramp-up"
                            self.stats['aborted_at_file'] = i + 1
                            self.stats['abort_time'] = overall_elapsed
                            self.stats['ramp_up_completed'] = True
                            
                            break
                    else:
                        # Reset slow detection counter if we're back to normal speed
                        if slow_detections > 0:
                            print(f"✅ Speed recovered: {window_rate:.1f}/s (was slow for {slow_detections} windows)")
                        slow_detections = 0
                        status_emoji = "🔥" if window_rate > 300 else "⚡"
                        print(f"{status_emoji} Progress: {i+1}/{len(image_files)} | "
                              f"Rate: {window_rate:.1f}/s | Overall: {overall_rate:.1f}/s | "
                              f"Parse: {avg_parse_time:.1f}ms | ETA: {eta:.0f}s")
                
                # Reset window timer - THIS MUST BE AFTER ALL THE CHECKS
                last_100_start = current_time
        
        # Quick progress updates during ramp-up for slower iterations
        if not ramp_up_completed and i % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            progress_to_target = (rate / ramp_up_target_rate) * 100
            print(f"⏳ Warming up: {rate:.1f}/s ({progress_to_target:.0f}% of {ramp_up_target_rate}/s target) - file {i+1}")
        
        elapsed = time.time() - start_time
        
        # Final summary
        if self.stats.get('abort_reason'):
            print(f"🚨 SCAN ABORTED: {self.stats['abort_reason']}")
            print(f"📊 Processed {self.stats['scanned_files']} files before abort")
            if ramp_up_completed:
                print(f"✅ Performance monitoring was active (ramp-up completed)")
            else:
                print(f"⚠️  Aborted during ramp-up phase (never reached {ramp_up_target_rate}/s)")
        else:
            print(f"✅ Scan complete! Processed {len(image_files)} files in {elapsed:.1f}s")
            final_rate = len(image_files) / elapsed if elapsed > 0 else 0
            print(f"📊 Final processing rate: {final_rate:.1f} files/second")
            if not ramp_up_completed:
                print(f"ℹ️  Note: Never reached ramp-up target of {ramp_up_target_rate}/s")
        
        # Store ramp-up info in stats
        self.stats['ramp_up_completed'] = ramp_up_completed
        self.stats['ramp_up_target'] = ramp_up_target_rate
        
        return unparsed_files

    def save_limited_directories_report(self, output_file: str = None):
        """Save a separate report of directories that were limited"""
        if not self.stats['limited_directories']:
            return None
        
        if output_file is None:
            output_file = self.generate_unique_filename("limited_directories_report", ".json")
        
        report = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "total_limited_directories": len(self.stats['limited_directories']),
                "total_skipped_files": sum(ld['skipped_files'] for ld in self.stats['limited_directories'])
            },
            "limited_directories": self.stats['limited_directories'],
            "directory_file_counts": self.stats['directory_limits']
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📂 Limited directories report saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Failed to save limited directories report: {e}")
            return None

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
                "scan_completed": not bool(self.stats.get('abort_reason')),
                "abort_reason": self.stats.get('abort_reason'),
                "aborted_at_file": self.stats.get('aborted_at_file'),
                "files_with_unparsed_metadata": len(unparsed_files),
                "likely_ai_generated": self.stats['ai_generated'],
                "scan_errors": self.stats['errors'],
                # New directory limiting info
                "directory_limiting": {
                    "limited_directories_count": len(self.stats['limited_directories']),
                    "total_files_skipped_by_limits": sum(ld['skipped_files'] for ld in self.stats['limited_directories']) if self.stats['limited_directories'] else 0,
                    "directory_file_counts": self.stats['directory_limits']
                }
            },
            "performance_stats": performance_stats,
            "directory_limits_applied": self.stats['limited_directories'],  # New section
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
        
        # Save main report
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
        
        # Save separate limited directories report if any directories were limited
        if self.stats['limited_directories']:
            limited_dirs_file = self.save_limited_directories_report()
        
        # Print summary with directory limiting info
        print(f"\n📊 SCAN RESULTS:")
        if self.stats.get('abort_reason'):
            print(f"🚨 ABORTED: {self.stats['abort_reason']}")
            print(f"📍 Stopped at file: {self.stats['aborted_at_file']}/{self.stats['total_files']}")
            print(f"⏱️  Time before abort: {self.stats.get('abort_time', 0):.1f}s")
        
        print(f"📁 Total files found: {self.stats['total_files']:,}")
        if self.stats['limited_directories']:
            total_skipped_by_limits = sum(ld['skipped_files'] for ld in self.stats['limited_directories'])
            print(f"📂 Limited {len(self.stats['limited_directories'])} directories, skipped {total_skipped_by_limits:,} files")
        print(f"🔍 Files actually scanned: {self.stats['scanned_files']:,}")
        print(f"📝 Files with unparsed metadata: {len(unparsed_files):,}")
        print(f"🤖 Likely AI-generated: {self.stats['ai_generated']:,}")
        print(f"❌ Scan errors: {self.stats['errors']:,}")
        
        # Show limited directories
        if self.stats['limited_directories']:
            print(f"\n📂 DIRECTORIES LIMITED:")
            for i, dir_info in enumerate(self.stats['limited_directories'][:10]):
                rel_path = os.path.relpath(dir_info['directory'])
                print(f"  {i+1}. {rel_path}")
                print(f"     📊 {dir_info['total_files']} files found, {dir_info['selected_files']} scanned, {dir_info['skipped_files']} skipped")
            
            if len(self.stats['limited_directories']) > 10:
                print(f"  ... and {len(self.stats['limited_directories']) - 10} more directories")
        
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
    parser.add_argument("--max-files-per-dir", "-d", type=int, 
                       help="Maximum files to scan per subdirectory")
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
        min_size_kb=args.min_size,
        max_files_per_dir=args.max_files_per_dir
    )
    
    scanner.generate_report(unparsed_files, args.output, args.min_size)

if __name__ == "__main__":
    main()