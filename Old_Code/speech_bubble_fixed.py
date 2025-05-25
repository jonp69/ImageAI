# speech_bubble_detector.py

import cv2
import numpy as np
import os
from typing import List, Dict

def detect_speech_bubbles(image_path: str, output_path: str = None, draw_bounding: bool = True) -> List[Dict]:
    """
    Detect speech bubbles in comic images.
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the annotated image
        draw_bounding: Whether to draw bounding boxes around detected bubbles
        
    Returns:
        List of dictionaries containing the bounding box coordinates of detected bubbles
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold and morphological operations to isolate bubbles
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        area = w * h

        # Filter by size and aspect ratio to find likely speech bubbles
        if area > 1000 and 0.3 < aspect_ratio < 3:
            bubble_regions.append({"x": x, "y": y, "w": w, "h": h})
            if draw_bounding:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if output_path:
        cv2.imwrite(output_path, image)

    return bubble_regions

def overwrite_speech_bubbles(image_path: str, output_path: str, bubble_regions: List[Dict] = None) -> None:
    """
    Overwrite detected speech bubbles with white polygons.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the modified image
        bubble_regions: List of bubble regions (if None, will detect them)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # If no regions provided, detect them
    if bubble_regions is None:
        bubble_regions = detect_speech_bubbles(image_path, draw_bounding=False)
    
    # Fill each bubble region with white
    for region in bubble_regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # -1 fills the rectangle
    
    # Save the modified image
    cv2.imwrite(output_path, image)
    
    return len(bubble_regions)