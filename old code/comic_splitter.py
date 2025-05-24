# comic_splitter.py

import cv2
import os
from PIL import Image
import numpy as np
import pytesseract
from typing import List, Tuple

def split_panels(image_path: str, output_dir: str, remove_text: bool = False) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panel_paths = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 10000:
            continue  # skip small regions

        panel = image[y:y+h, x:x+w]

        if remove_text:
            panel = remove_text_from_image(panel)

        panel_filename = f"{basename}_panel_{i}.png"
        panel_path = os.path.join(output_dir, panel_filename)
        cv2.imwrite(panel_path, panel)
        panel_paths.append(panel_path)

    return panel_paths

def remove_text_from_image(image_np: np.ndarray) -> np.ndarray:
    d = pytesseract.image_to_data(image_np, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if w > 0 and h > 0:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return image_np
