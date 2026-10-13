"""
backend.vision: Slide Keyframe Scene Detection, OCR, and Vietnamese Diacritics Recovery.
"""
from backend.vision.vision_extractor import vision_extractor, VisionExtractor
from backend.vision.vietnamese_text_enhancer import VietnameseTextEnhancer, remove_accents

__all__ = [
    "vision_extractor",
    "VisionExtractor",
    "VietnameseTextEnhancer",
    "remove_accents",
]
