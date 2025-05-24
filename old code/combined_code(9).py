
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict


def hash_file(filepath: str) -> str:
    """Generate a hash for a given file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def find_images(base_dir: str, extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"]) -> List[str]:
    """
    Recursively find all image files in a base directory with the given extensions.

    Args:
        base_dir: Directory to search for images.
        extensions: List of allowed file extensions.

    Returns:
        A list of file paths for the discovered images.
    """
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(base_dir).rglob(f"*{ext}"))
    return [str(p.resolve()) for p in image_paths]


def load_metadata(
    metadata_path="image_metadata.json", blacklist_path="blacklist.json", image_dir="images/"
) -> Dict[str, dict]:
    """
    Load metadata from a file. If the metadata file doesn't exist and there are images in the directory, 
    the metadata file is created. If no images are found, raises an error.

    Args:
        metadata_path: Path to the metadata file.
        blacklist_path: Path to the blacklist file.
        image_dir: Directory where image files are stored.

    Returns:
        Metadata dictionary.
    """
    if os.path.exists(metadata_path):
        # Load existing metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Find images in the directory
    images = find_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in '{image_dir}'. Cannot create metadata file.")

    # Load blacklist if it exists
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = json.load(f)
    else:
        blacklist = {}

    # Create metadata for images not in the blacklist
    metadata = {}
    for img_path in images:
        img_hash = hash_file(img_path)
        if img_hash not in blacklist:
            metadata[img_path] = {
                "name": os.path.basename(img_path),
                "hash": img_hash,
                "predicted_tags": {},  # Placeholder for tags, will be populated later
            }

    # Save the generated metadata to a file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Created metadata file at '{metadata_path}' with {len(metadata)} entries.")

    return metadata


def update_blacklist(image_path: str, blacklist_path="blacklist.json"):
    """
    Blacklist a given image by adding its hash and name to the blacklist file.

    Args:
        image_path: Path to the image to blacklist.
        blacklist_path: Path to the blacklist file.
    """
    img_hash = hash_file(image_path)
    blacklist = {}

    # Load existing blacklist if it exists
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r", encoding="utf-8") as f:
            blacklist = json.load(f)

    # Add the image to the blacklist
    blacklist[img_hash] = {"name": os.path.basename(image_path)}

    # Save the updated blacklist to the file
    with open(blacklist_path, "w", encoding="utf-8") as f:
        json.dump(blacklist, f, indent=2, ensure_ascii=False)
    print(f"Image '{image_path}' has been added to the blacklist.")


def remove_from_metadata(image_path: str, metadata_path="image_metadata.json", blacklist_path="blacklist.json"):
    """
    Remove an image from the metadata file and add it to the blacklist.

    Args:
        image_path: Path to the image to remove from metadata.
        metadata_path: Path to the metadata file.
        blacklist_path: Path to the blacklist file.
    """
    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata file found at '{metadata_path}'.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Check if the image exists in metadata
    if image_path in metadata:
        # Remove the image from metadata
        del metadata[image_path]

        # Update metadata file
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Add the image to the blacklist
        update_blacklist(image_path, blacklist_path)
    else:
        print(f"Image '{image_path}' not found in metadata.")

