"""
ISL-Malayalam Translation System
Main application entry point for Indian Sign Language to Malayalam text and speech translation.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.sign_recognition.detector import SignLanguageDetector
from src.text_processing.malayalam_processor import MalayalamProcessor
from src.speech_synthesis.tts_engine import TextToSpeechEngine
from src.gui.main_window import ISLTranslatorGUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    try:
        logger.info("Starting ISL-Malayalam Translation System")

        # Initialize components
        sign_detector = SignLanguageDetector()
        malayalam_processor = MalayalamProcessor()
        tts_engine = TextToSpeechEngine()

        # Launch GUI
        app = ISLTranslatorGUI(
            sign_detector=sign_detector,
            text_processor=malayalam_processor,
            tts_engine=tts_engine,
        )

        app.run()

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
