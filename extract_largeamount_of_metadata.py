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
            'parsing_times': [],
            'directory_limits': {},
            'limited_directories': [],
            'performance_limited_directories': []  # New: directories limited due to performance
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
    
    def detect_slow_directory(self, recent_files: list, performance_window: int = 50) -> dict:
        """Detect if recent files indicate a slow directory pattern"""
        if len(recent_files) < performance_window:
            return {'is_slow': False}
        
        # Get the last N files
        recent_batch = recent_files[-performance_window:]
        
        # Check if they're mostly from the same directory
        directories = defaultdict(int)
        slow_files = 0
        
        for file_info in recent_batch:
            file_dir = os.path.dirname(file_info['filename']) if '/' in file_info['filename'] or '\\' in file_info['filename'] else 'root'
            directories[file_dir] += 1
            
            # Consider file slow if it's 3x+ the median
            if file_info['parse_time_ms'] > 30:  # Arbitrary threshold for "slow"
                slow_files += 1
        
        # Find the dominant directory
        dominant_dir = max(directories.items(), key=lambda x: x[1])
        dominant_dir_name, file_count = dominant_dir
        
        # If 70%+ of recent files are from same directory and 40%+ are slow
        if file_count >= performance_window * 0.7 and slow_files >= performance_window * 0.4:
            avg_time = sum(f['parse_time_ms'] for f in recent_batch) / len(recent_batch)
            return {
                'is_slow': True,
                'directory': dominant_dir_name,
                'slow_file_ratio': slow_files / performance_window,
                'directory_file_ratio': file_count / performance_window,
                'avg_parse_time': avg_time,
                'files_in_batch': file_count
            }
        
        return {'is_slow': False}

    def scan_directory(self, directory: str, recursive: bool = True, 
        extensions: list = None, max_files: int = None,
        min_size_kb: int = 50, slowdown_threshold: int = 50) -> list:
        """Scan directory for images with unparsed metadata with dynamic performance limiting"""
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']
        
        print(f"🔍 Scanning directory: {directory}")
        print(f"📁 Recursive: {recursive}")
        print(f"📄 Extensions: {extensions}")
        print(f"📏 Min file size: {min_size_kb}KB ({min_size_kb * 1024:,} bytes)")
        print(f"🐌 Slowdown detection window: {slowdown_threshold} files")
        print(f"🚀 Will ramp up to 300+/sec before monitoring for slowdowns below 200/sec")
        
        # Performance monitoring variables
        performance_window = 100  # Check every 100 files
        min_acceptable_rate = 200.0  # files per second
        ramp_up_target_rate = 300.0  # Target rate to reach before monitoring
        ramp_up_completed = False
        slow_detections = 0
        max_slow_detections = 10  # Allow more slow periods before aborting (since we're blocking directories)
        
        # Calculate individual file performance threshold
        max_acceptable_time_ms = (1.0 / min_acceptable_rate) * 1000  # Convert to milliseconds
        print(f"📏 Individual file threshold: {max_acceptable_time_ms:.1f}ms per file (for {min_acceptable_rate}/s rate)")
        
        # Dynamic directory foul tracking
        directory_file_counts = defaultdict(int)
        blocked_directories = set()  # Directories to skip due to performance
        
        # New: Dynamic directory foul system
        directory_foul_counts = defaultdict(int)  # Track fouls per directory
        recent_files_window = []  # Rolling window of recent file processing info
        foul_threshold_per_directory = slowdown_threshold // 4  # Block after N fouls (25% of window)
        
        print(f"⚠️  Directory blocking: Will block directories with {foul_threshold_per_directory}+ fouls in {slowdown_threshold}-file window")
        
        # Collect all image files
        image_files = []
        skipped_small = 0
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip directories that have been blocked for performance
                if root in blocked_directories:
                    continue
                
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        
                        # Check if this directory has been performance-limited
                        if root in blocked_directories:
                            continue
                        
                        # Check file size
                        if self.is_file_large_enough(file_path, min_size_kb):
                            image_files.append(file_path)
                            directory_file_counts[root] += 1
                            
                            # Check global max_files limit
                            if max_files and len(image_files) >= max_files:
                                break
                        else:
                            skipped_small += 1
                
                if max_files and len(image_files) >= max_files:
                    break
        else:
            # Non-recursive: scan main directory only
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in extensions):
                    # Check file size
                    if self.is_file_large_enough(file_path, min_size_kb):
                        image_files.append(file_path)
                        directory_file_counts[directory] += 1
                        
                        # Check global max_files limit
                        if max_files and len(image_files) >= max_files:
                            break
                    else:
                        skipped_small += 1
        
        # Store initial stats
        self.stats['total_files'] = len(image_files) + skipped_small
        self.stats['skipped_small_files'] = skipped_small
        self.stats['directory_limits'] = dict(directory_file_counts)
        
        print(f"📊 Found {len(image_files) + skipped_small:,} total image files")
        print(f"📏 Skipped {skipped_small:,} files under {min_size_kb}KB")
        print(f"🔍 Scanning {len(image_files):,} files")
        
        # Scan files with dynamic performance monitoring
        unparsed_files = []
        start_time = time.time()
        last_100_start = start_time
        
        i = 0
        while i < len(image_files):
            current_time = time.time()
            image_path = image_files[i]
            current_dir = os.path.dirname(image_path)
            
            # Skip if directory has been performance-blocked
            if current_dir in blocked_directories:
                i += 1
                continue
            
            result = self.scan_image_fast(image_path)
            self.stats['scanned_files'] += 1
            
            # Track file in rolling window
            file_record = {
                'filename': os.path.basename(image_path),
                'directory': current_dir,
                'parse_time_ms': result['parse_time_ms'],
                'file_size': result['file_size'],
                'index': i,
                'is_foul': result['parse_time_ms'] > max_acceptable_time_ms
            }
            recent_files_window.append(file_record)
            
            # Maintain rolling window size
            if len(recent_files_window) > slowdown_threshold:
                # Remove oldest record and subtract any fouls from directory counts
                oldest_record = recent_files_window.pop(0)
                if oldest_record['is_foul']:
                    directory_foul_counts[oldest_record['directory']] -= 1
                    if directory_foul_counts[oldest_record['directory']] <= 0:
                        del directory_foul_counts[oldest_record['directory']]
            
            # Add foul to directory count if this file was slow
            if file_record['is_foul']:
                directory_foul_counts[current_dir] += 1
            
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
                    # Ramp-up completed - now monitor for slowdowns and directory fouls
                    if window_rate < min_acceptable_rate:
                        slow_detections += 1
                        
                        # Show current foul status for directories
                        current_fouls = {d: count for d, count in directory_foul_counts.items() if count > 0}
                        foul_summary = ", ".join([f"{os.path.basename(d)}:{count}" for d, count in sorted(current_fouls.items(), key=lambda x: x[1], reverse=True)[:3]])
                        
                        print(f"🐌 SLOW WINDOW {slow_detections}/{max_slow_detections}: {i+1}/{len(image_files)} | "
                              f"Rate: {window_rate:.1f}/s (target: {min_acceptable_rate}/s) | "
                              f"Avg parse: {avg_parse_time:.1f}ms | ETA: {eta:.0f}s")
                        
                        if foul_summary:
                            print(f"   ⚠️  Directory fouls: {foul_summary}")
                        
                        # Check for directories that need blocking
                        directories_to_block = []
                        for dir_path, foul_count in directory_foul_counts.items():
                            if foul_count >= foul_threshold_per_directory and dir_path not in blocked_directories:
                                directories_to_block.append((dir_path, foul_count))
                        
                        # Block problematic directories
                        for dir_path, foul_count in directories_to_block:
                            blocked_directories.add(dir_path)
                            
                            # Count remaining files in this directory
                            remaining_files = sum(1 for path in image_files[i+1:] 
                                                if os.path.dirname(path) == dir_path)
                            
                            # Calculate statistics for this directory
                            dir_files_in_window = [f for f in recent_files_window if f['directory'] == dir_path]
                            avg_dir_time = sum(f['parse_time_ms'] for f in dir_files_in_window) / len(dir_files_in_window) if dir_files_in_window else 0
                            
                            self.stats['performance_limited_directories'].append({
                                'directory': dir_path,
                                'foul_count': foul_count,
                                'files_in_window': len(dir_files_in_window),
                                'avg_parse_time': avg_dir_time,
                                'files_processed_before_block': directory_file_counts[dir_path] - remaining_files,
                                'files_blocked': remaining_files,
                                'blocked_at_file_index': i + 1,
                                'threshold_exceeded': f"{foul_count}/{foul_threshold_per_directory} fouls"
                            })
                            
                            print(f"   🚫 BLOCKING DIRECTORY: {os.path.relpath(dir_path)}")
                            print(f"   📊 Fouls: {foul_count}/{foul_threshold_per_directory} | "
                                  f"Files in window: {len(dir_files_in_window)} | "
                                  f"Avg time: {avg_dir_time:.1f}ms")
                            print(f"   📂 Skipping {remaining_files} remaining files in this directory")
                            
                            # Reset slow detection counter since we took action
                            slow_detections = max(0, slow_detections - 1)
                        
                        # Abort after too many consecutive slow periods without resolution
                        if slow_detections >= max_slow_detections:
                            print(f"\n🚨 ABORTING: {slow_detections} consecutive slow periods detected!")
                            print(f"📊 Processed {i+1} files in {overall_elapsed:.1f}s before aborting")
                            print(f"🐌 Current rate: {window_rate:.1f} files/second (target: {min_acceptable_rate})")
                            print(f"🚫 Blocked {len(blocked_directories)} directories")
                            
                            # Store abort reason in stats
                            self.stats['abort_reason'] = f"Processing rate remained below {min_acceptable_rate} files/sec despite blocking {len(blocked_directories)} directories"
                            self.stats['aborted_at_file'] = i + 1
                            self.stats['abort_time'] = overall_elapsed
                            self.stats['ramp_up_completed'] = ramp_up_completed
                            
                            break
                    else:
                        # Reset slow detection counter if we're back to normal speed
                        if slow_detections > 0:
                            print(f"✅ Speed recovered: {window_rate:.1f}/s (was slow for {slow_detections} windows)")
                        slow_detections = 0
                        
                        # Show foul status if there are active fouls
                        if directory_foul_counts:
                            active_fouls = sum(directory_foul_counts.values())
                            status_emoji = "🔥" if window_rate > 300 else "⚡"
                            print(f"{status_emoji} Progress: {i+1}/{len(image_files)} | "
                                  f"Rate: {window_rate:.1f}/s | Overall: {overall_rate:.1f}/s | "
                                  f"Parse: {avg_parse_time:.1f}ms | Active fouls: {active_fouls} | ETA: {eta:.0f}s")
                        else:
                            status_emoji = "🔥" if window_rate > 300 else "⚡"
                            print(f"{status_emoji} Progress: {i+1}/{len(image_files)} | "
                                  f"Rate: {window_rate:.1f}/s | Overall: {overall_rate:.1f}/s | "
                                  f"Parse: {avg_parse_time:.1f}ms | ETA: {eta:.0f}s")
                
                # Reset window timer
                last_100_start = current_time
            
            # Quick progress updates during ramp-up for slower iterations
            elif not ramp_up_completed and i % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                progress_to_target = (rate / ramp_up_target_rate) * 100
                print(f"⏳ Warming up: {rate:.1f}/s ({progress_to_target:.0f}% of {ramp_up_target_rate}/s target) - file {i+1}")
            
            i += 1
        
        elapsed = time.time() - start_time
        
        # Update stats with performance blocking info
        if blocked_directories:
            total_blocked_files = sum(ld['files_blocked'] for ld in self.stats['performance_limited_directories'])
            print(f"\n🚫 PERFORMANCE BLOCKING SUMMARY:")
            print(f"📂 Blocked {len(blocked_directories)} directories due to slowdowns")
            print(f"📊 Skipped {total_blocked_files} files in blocked directories")
            
            print(f"\n📊 DIRECTORY FOUL DETAILS:")
            for dir_info in self.stats['performance_limited_directories']:
                rel_path = os.path.relpath(dir_info['directory'])
                print(f"  🚫 {rel_path}")
                print(f"     ⚠️  {dir_info['threshold_exceeded']} | Avg time: {dir_info['avg_parse_time']:.1f}ms")
                print(f"     📂 {dir_info['files_processed_before_block']} processed, {dir_info['files_blocked']} skipped")
        
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
                )[:20],
                "file_size_distribution": {
                    "total_files": len(unparsed_files),
                    "average_size_kb": sum(f['file_size'] for f in unparsed_files) / 1024 / len(unparsed_files) if unparsed_files else 0,
                    "size_kb_percentiles": {
                        "p25": sorted(f['file_size'] for f in unparsed_files)[len(unparsed_files) // 4] / 1024 if unparsed_files else 0,
                        "p50": sorted(f['file_size'] for f in unparsed_files)[len(unparsed_files) // 2] / 1024 if unparsed_files else 0,
                        "p75": sorted(f['file_size'] for f in unparsed_files)[3 * len(unparsed_files) // 4] / 1024 if unparsed_files else 0
                    }
                }
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

    def generate_performance_report(self):
        """Generate performance troubleshooting report"""
        if not self.stats['parsing_times']:
            return
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        
        # Calculate statistics
        all_times = [t['parse_time_ms'] for t in self.stats['parsing_times']]
        all_times.sort()
        
        avg_time = sum(all_times) / len(all_times)
        median_time = all_times[len(all_times) // 2]
        min_time = min(all_times)
        max_time = max(all_times)
        
        print(f"📊 Parsing Time Statistics:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Median:  {median_time:.2f}ms")
        print(f"  Fastest: {min_time:.2f}ms")
        print(f"  Slowest: {max_time:.2f}ms")
        
        # Format breakdown by format
        format_times = defaultdict(list)
        for timing in self.stats['parsing_times']:
            format_times[timing['format']].append(timing['parse_time_ms'])
        
        print(f"\n📁 Average Time by Format:")
        for format_name, times in format_times.items():
            avg_format_time = sum(times) / len(times)
            print(f"  {format_name}: {avg_format_time:.2f}ms ({len(times)} files)")
        
        # Show slowest files
        slowest_files = sorted(self.stats['parsing_times'], 
                             key=lambda x: x['parse_time_ms'], reverse=True)[:10]
        
        print(f"\n🐌 TOP 10 SLOWEST FILES (Troubleshooting):")
        print("=" * 80)
        
        for i, file_info in enumerate(slowest_files, 1):
            filename = os.path.basename(file_info['filename'])
            parse_time = file_info['parse_time_ms']
            file_size_kb = file_info['file_size'] / 1024
            format_name = file_info['format']
            chunks_found = file_info['chunks_found']
            has_error = file_info['has_error']
            
            print(f"\n{i}. {filename}")
            print(f"   ⏱️  Total time: {parse_time:.2f}ms")
            print(f"   📏 File size: {file_size_kb:.1f}KB")
            print(f"   📄 Format: {format_name}")
            print(f"   🔍 Chunks found: {chunks_found}")
            print(f"   ❌ Has error: {has_error}")
            
            # Show performance breakdown if available
            if 'performance_breakdown' in file_info:
                breakdown = file_info['performance_breakdown']
                print(f"   📊 Time breakdown:")
                print(f"      File access: {breakdown.get('file_access', 0):.2f}ms")
                print(f"      Image open:  {breakdown.get('image_open', 0):.2f}ms")
                print(f"      PNG chunks:  {breakdown.get('png_chunks', 0):.2f}ms")
                print(f"      EXIF data:   {breakdown.get('exif_data', 0):.2f}ms")
                print(f"      Analysis:    {breakdown.get('metadata_analysis', 0):.2f}ms")
            
            print(f"   📈 Efficiency: {parse_time/file_size_kb:.2f}ms/KB")
            
            # Identify potential issues
            issues = []
            if parse_time > avg_time * 3:
                issues.append("Extremely slow (3x+ average)")
            if file_size_kb > 5000:
                issues.append("Very large file")
            if breakdown.get('image_open', 0) > 50:
                issues.append("Slow image opening (corrupt/large file?)")
            if breakdown.get('png_chunks', 0) > 50:
                issues.append("Slow PNG chunk processing")
            
            if issues:
                print(f"   ⚠️  Potential issues: {', '.join(issues)}")
        
        # Analyze slow file patterns
        print(f"\n🔍 SLOW FILE PATTERNS:")
        slow_files = [f for f in self.stats['parsing_times'] if f['parse_time_ms'] > avg_time * 2]
        
        if slow_files:
            slow_formats = defaultdict(int)
            for f in slow_files[-20:]:  # Last 20 slow files
                slow_formats[f['format']] += 1
            
            print(f"📄 Formats in slowest 20:")
            for format_name, count in slow_formats.items():
                print(f"   {format_name}: {count} files")
            
            avg_slow_size = sum(f['file_size'] for f in slow_files[-20:]) / len(slow_files[-20:]) / 1024
            print(f"📏 Average size of slowest 20: {avg_slow_size:.1f}KB")
    
def main():
    parser = argparse.ArgumentParser(description="Scan large image datasets for unparsed metadata")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan recursively")
    parser.add_argument("--max-files", "-m", type=int, help="Maximum files to scan")
    parser.add_argument("--slowdown-threshold", "-t", type=int, default=50,
                       help="Number of recent files to analyze for directory slowdowns (default: 50)")
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
        slowdown_threshold=args.slowdown_threshold
    )
    
    scanner.generate_report(unparsed_files, args.output, args.min_size)

if __name__ == "__main__":
    main()