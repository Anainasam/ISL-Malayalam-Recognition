"""
Simple test script to verify that all components work correctly.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")

    try:
        import cv2

        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False

    try:
        import mediapipe as mp

        print(f"✓ MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False

    try:
        import pyttsx3

        print("✓ pyttsx3 imported successfully")
    except ImportError as e:
        print(f"✗ pyttsx3 import failed: {e}")
        return False

    try:
        import customtkinter as ctk

        print("✓ CustomTkinter imported successfully")
    except ImportError as e:
        print(f"✗ CustomTkinter import failed: {e}")
        return False

    try:
        from gtts import gTTS

        print("✓ gTTS imported successfully")
    except ImportError as e:
        print(f"✗ gTTS import failed: {e}")
        return False

    try:
        import pygame

        print(f"✓ Pygame version: {pygame.version.ver}")
    except ImportError as e:
        print(f"✗ Pygame import failed: {e}")
        return False

    return True


def test_components():
    """Test individual components."""
    print("\nTesting components...")

    try:
        from src.sign_recognition.detector import SignLanguageDetector

        detector = SignLanguageDetector()
        print("✓ SignLanguageDetector created successfully")
    except Exception as e:
        print(f"✗ SignLanguageDetector failed: {e}")
        return False

    try:
        from src.text_processing.malayalam_processor import MalayalamProcessor

        processor = MalayalamProcessor()
        print("✓ MalayalamProcessor created successfully")

        # Test basic text processing
        test_text = "നമസ്കാരം"
        cleaned = processor.clean_text(test_text)
        is_malayalam = processor.is_malayalam_text(test_text)
        print(
            f"✓ Text processing test: '{test_text}' -> '{cleaned}', Malayalam: {is_malayalam}"
        )

    except Exception as e:
        print(f"✗ MalayalamProcessor failed: {e}")
        return False

    try:
        from src.speech_synthesis.tts_engine import TextToSpeechEngine

        tts = TextToSpeechEngine()
        info = tts.get_engine_info()
        print(f"✓ TextToSpeechEngine created successfully")
        print(f"  Engine info: {info}")
    except Exception as e:
        print(f"✗ TextToSpeechEngine failed: {e}")
        return False

    return True


def test_camera():
    """Test camera availability."""
    print("\nTesting camera availability...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera 0 is available")
            cap.release()
            return True
        else:
            print("✗ Camera 0 is not available")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== ISL-Malayalam Translation System Test ===\n")

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test components
    if not test_components():
        success = False

    # Test camera
    if not test_camera():
        print(
            "  Note: Camera test failed, but this is expected if no camera is connected"
        )

    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed! The system is ready to use.")
        print("\nTo run the application:")
        print("  python main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
