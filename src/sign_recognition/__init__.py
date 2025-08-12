                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # Sign Recognition Module
"""
ISL Sign Recognition Module
Provides real-time sign language recognition capabilities.
"""

__version__ = "1.0.0"
__author__ = "ISL-Malayalam Project"

from .detector import SignLanguageDetector
from .preprocessor import VideoPreprocessor

__all__ = ["SignLanguageDetector", "VideoPreprocessor"]
