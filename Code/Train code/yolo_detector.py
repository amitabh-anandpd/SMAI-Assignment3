"""
YOLO License Plate Detection Module
Handles YOLO model loading and plate detection inference.
"""

from ultralytics import YOLO
from typing import List, Tuple
import numpy as np


class YOLODetector:
    """Wrapper for YOLO license plate detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path (str): Path to the trained YOLO model (.pt file).
        """
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("✓ YOLO model loaded successfully!")
    
    def detect_plates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plates in an image.
        
        Args:
            image (np.ndarray): Input image in RGB format.
        
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes [(x1, y1, x2, y2), ...].
                                              Empty list if no plates detected.
        """
        results = self.model(image, verbose=False)
        
        bboxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def detect_and_crop(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect license plates and crop them from the image.
        
        Args:
            image (np.ndarray): Input image in RGB format.
        
        Returns:
            List[Tuple[np.ndarray, Tuple]]: List of (cropped_image, bbox) pairs.
                                            Empty list if no plates detected.
        """
        bboxes = self.detect_plates(image)
        cropped_plates = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]
            cropped_plates.append((cropped, bbox))
        
        return cropped_plates
