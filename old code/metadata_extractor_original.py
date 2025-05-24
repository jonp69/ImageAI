from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

def extract_image_metadata(image_path: str) -> dict:
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            meta = {
                "resolution": f"{img.width}x{img.height}",
                "format": img.format,
                "mode": img.mode,
                "prompt": None,
                "seed": None,
                "sampler": None,
                "source_tags": {},
            }

            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (int, float, str)):
                        meta[tag] = value
                        if tag == "Prompt":
                            meta["prompt"] = value
                        elif tag == "Seed":
                            try:
                                meta["seed"] = int(value)
                            except ValueError:
                                pass
                        elif tag == "Sampler":
                            meta["sampler"] = value

            return meta
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        return {}

def find_images(base_dir: str) -> list:
    image_extensions = (".png", ".jpg", ".jpeg", ".webp")
    return [p for p in Path(base_dir).rglob("*") if p.suffix.lower() in image_extensions]