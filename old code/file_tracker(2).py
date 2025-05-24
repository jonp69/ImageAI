# file_tracker.py

import os
import json
from typing import Dict, Set, List

class FileTracker:
    def __init__(self, metadata_path: str = "image_metadata.json"):
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, dict]:
        """Load existing metadata or create new metadata file if none exists."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {self.metadata_path}, creating new metadata")
                return {}
        else:
            return {}
    
    def save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def update_file_status(self, base_dirs: List[str], extensions: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}):
        """
        Update file status flags:
        - Mark files as available (exists=True) if they exist
        - Mark files as deleted (exists=False) if they don't exist but are in metadata
        - Identify duplicates based on file hash or other properties
        
        Args:
            base_dirs: List of directories to scan for images
            extensions: Set of file extensions to consider
        """
        # Track existing files to identify what's been deleted
        existing_files = set()
        
        # Scan all directories for image files
        for base_dir in base_dirs:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        full_path = os.path.normpath(os.path.join(root, file))
                        existing_files.add(full_path)
                        
                        # Update or add metadata for this file
                        if full_path in self.metadata:
                            self.metadata[full_path]["exists"] = True
                        else:
                            # New file, add basic entry
                            self.metadata[full_path] = {
                                "exists": True,
                                "last_modified": os.path.getmtime(full_path),
                                "size": os.path.getsize(full_path)
                            }
        
        # Mark files that no longer exist as deleted, but keep their metadata
        for path in list(self.metadata.keys()):
            if path not in existing_files:
                self.metadata[path]["exists"] = False
        
        # Save updated metadata
        self.save_metadata()
        
        return {
            "total_files": len(self.metadata),
            "existing_files": len(existing_files),
            "deleted_files": len(self.metadata) - len(existing_files)
        }
    
    def find_duplicates(self, check_method: str = "path") -> Dict[str, List[str]]:
        """
        Find potential duplicate images using different methods.
        
        Args:
            check_method: Method to identify duplicates:
                - "path": Check if filename appears in multiple locations
                - "size": Group by identical file size
                - "metadata": Group by identical metadata parameters
                
        Returns:
            Dictionary of duplicate groups
        """
        duplicates = {}
        
        if check_method == "path":
            # Group by filename regardless of directory
            by_name = {}
            for path in self.metadata:
                filename = os.path.basename(path)
                by_name.setdefault(filename, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {name: paths for name, paths in by_name.items() if len(paths) > 1}
            
        elif check_method == "size":
            # Group by file size
            by_size = {}
            for path, info in self.metadata.items():
                if info.get("exists", True) and "size" in info:
                    size = info["size"]
                    by_size.setdefault(size, []).append(path)
            
            # Keep only groups with multiple entries
            duplicates = {f"size_{size}": paths for size, paths in by_size.items() if len(paths) > 1}
            
        elif check_method == "metadata":
            # Group by metadata fingerprint (resolution + format + seed if available)
            by_meta = {}
            for path, info in self.metadata.items():
                if info.get("exists", True):
                    # Create a metadata fingerprint
                    resolution = info.get("resolution", "")
                    img_format = info.get("format", "")
                    seed = info.get("seed", "")
                    fingerprint = f"{resolution}_{img_format}_{seed}"
                    
                    by_meta.setdefault(fingerprint, []).append(path)
            
            # Keep only groups with multiple entries and valid fingerprints
            duplicates = {meta: paths for meta, paths in by_meta.items() 
                         if len(paths) > 1 and meta != "__"}
        
        return duplicates