"""
Text-to-Speech Engine Module
Handles Malayalam text-to-speech synthesis using various TTS engines.
"""

import os
import logging
import threading
from typing import Optional, Dict, List
from pathlib import Path
import pyttsx3
import pygame
from gtts import gTTS
import tempfile

logger = logging.getLogger(__name__)


class TextToSpeechEngine:
    """
    Text-to-Speech engine for Malayalam text.
    Supports both offline (pyttsx3) and online (gTTS) synthesis.
    """

    def __init__(self, engine_type: str = "pyttsx3"):
        """
        Initialize TTS engine.

        Args:
            engine_type: Type of TTS engine ("pyttsx3" or "gtts")
        """
        self.engine_type = engine_type
        self.is_speaking = False
        self.speech_thread = None

        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
            self.pygame_available = True
        except Exception as e:
            logger.warning(f"Could not initialize pygame: {e}")
            self.pygame_available = False

        # Initialize pyttsx3 engine
        self.pyttsx3_engine = None
        if engine_type == "pyttsx3":
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self._configure_pyttsx3()
            except Exception as e:
                logger.error(f"Could not initialize pyttsx3: {e}")

        # Voice settings
        self.voice_settings = {
            "rate": 150,  # Words per minute
            "volume": 0.8,  # Volume level (0.0 to 1.0)
            "voice_id": None,  # Specific voice ID
        }

        logger.info(f"TextToSpeechEngine initialized with {engine_type}")

    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine settings."""
        if not self.pyttsx3_engine:
            return

        try:
            # Get available voices
            voices = self.pyttsx3_engine.getProperty("voices")

            # Try to find Malayalam or Hindi voice (closest to Malayalam)
            malayalam_voice = None
            hindi_voice = None

            for voice in voices:
                voice_name = voice.name.lower()
                if "malayalam" in voice_name or "ml" in voice_name:
                    malayalam_voice = voice.id
                    break
                elif "hindi" in voice_name or "hi" in voice_name:
                    hindi_voice = voice.id

            # Set voice preference
            if malayalam_voice:
                self.pyttsx3_engine.setProperty("voice", malayalam_voice)
                logger.info("Using Malayalam voice")
            elif hindi_voice:
                self.pyttsx3_engine.setProperty("voice", hindi_voice)
                logger.info("Using Hindi voice for Malayalam text")

            # Set speech rate and volume
            self.pyttsx3_engine.setProperty("rate", self.voice_settings["rate"])
            self.pyttsx3_engine.setProperty("volume", self.voice_settings["volume"])

        except Exception as e:
            logger.error(f"Error configuring pyttsx3: {e}")

    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available voices.

        Returns:
            List of voice dictionaries with id, name, and language info
        """
        voices = []

        if self.pyttsx3_engine:
            try:
                pyttsx3_voices = self.pyttsx3_engine.getProperty("voices")
                for voice in pyttsx3_voices:
                    voices.append(
                        {
                            "id": voice.id,
                            "name": voice.name,
                            "language": (
                                getattr(voice, "languages", ["unknown"])[0]
                                if hasattr(voice, "languages") and voice.languages
                                else "unknown"
                            ),
                            "engine": "pyttsx3",
                        }
                    )
            except Exception as e:
                logger.error(f"Error getting pyttsx3 voices: {e}")

        # Add gTTS options
        voices.extend(
            [
                {
                    "id": "gtts-ml",
                    "name": "Google TTS Malayalam",
                    "language": "ml",
                    "engine": "gtts",
                },
                {
                    "id": "gtts-hi",
                    "name": "Google TTS Hindi",
                    "language": "hi",
                    "engine": "gtts",
                },
                {
                    "id": "gtts-en",
                    "name": "Google TTS English",
                    "language": "en",
                    "engine": "gtts",
                },
            ]
        )

        return voices

    def set_voice_settings(
        self,
        rate: Optional[int] = None,
        volume: Optional[float] = None,
        voice_id: Optional[str] = None,
    ):
        """
        Set voice parameters.

        Args:
            rate: Speech rate in words per minute
            volume: Volume level (0.0 to 1.0)
            voice_id: Specific voice ID to use
        """
        if rate is not None:
            self.voice_settings["rate"] = max(50, min(300, rate))

        if volume is not None:
            self.voice_settings["volume"] = max(0.0, min(1.0, volume))

        if voice_id is not None:
            self.voice_settings["voice_id"] = voice_id

        # Apply settings to pyttsx3 engine
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.setProperty("rate", self.voice_settings["rate"])
                self.pyttsx3_engine.setProperty("volume", self.voice_settings["volume"])

                if voice_id:
                    self.pyttsx3_engine.setProperty("voice", voice_id)
            except Exception as e:
                logger.error(f"Error setting voice properties: {e}")

    def speak_text(self, text: str, blocking: bool = False) -> bool:
        """
        Convert text to speech.

        Args:
            text: Malayalam text to speak
            blocking: Whether to block until speech is complete

        Returns:
            True if speech started successfully, False otherwise
        """
        if not text or not text.strip():
            return False

        if self.is_speaking and not blocking:
            logger.warning("Speech already in progress")
            return False

        try:
            if blocking:
                return self._speak_blocking(text)
            else:
                return self._speak_async(text)
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            return False

    def _speak_blocking(self, text: str) -> bool:
        """Speak text in blocking mode."""
        if self.engine_type == "pyttsx3" and self.pyttsx3_engine:
            try:
                self.is_speaking = True
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                self.is_speaking = False
                return True
            except Exception as e:
                logger.error(f"pyttsx3 speaking error: {e}")
                self.is_speaking = False
                return False

        elif self.engine_type == "gtts":
            return self._speak_with_gtts(text)

        return False

    def _speak_async(self, text: str) -> bool:
        """Speak text asynchronously."""
        if self.speech_thread and self.speech_thread.is_alive():
            return False

        self.speech_thread = threading.Thread(target=self._speak_blocking, args=(text,))
        self.speech_thread.daemon = True
        self.speech_thread.start()
        return True

    def _speak_with_gtts(self, text: str) -> bool:
        """
        Speak text using Google Text-to-Speech.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        if not self.pygame_available:
            logger.error("pygame not available for gTTS playback")
            return False

        try:
            self.is_speaking = True

            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_filename = temp_file.name

            # Generate speech with gTTS
            tts = gTTS(text=text, lang="ml", slow=False)  # Malayalam language
            tts.save(temp_filename)

            # Play audio with pygame
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            # Clean up
            os.unlink(temp_filename)
            self.is_speaking = False
            return True

        except Exception as e:
            logger.error(f"gTTS error: {e}")
            self.is_speaking = False
            # Clean up temp file if it exists
            try:
                if "temp_filename" in locals():
                    os.unlink(temp_filename)
            except:
                pass
            return False

    def stop_speaking(self):
        """Stop current speech synthesis."""
        try:
            self.is_speaking = False

            if self.engine_type == "pyttsx3" and self.pyttsx3_engine:
                self.pyttsx3_engine.stop()

            elif self.engine_type == "gtts" and self.pygame_available:
                pygame.mixer.music.stop()

            logger.info("Speech stopped")

        except Exception as e:
            logger.error(f"Error stopping speech: {e}")

    def is_busy(self) -> bool:
        """
        Check if TTS engine is currently speaking.

        Returns:
            True if speaking, False otherwise
        """
        return self.is_speaking

    def save_speech_to_file(
        self, text: str, filename: str, language: str = "ml"
    ) -> bool:
        """
        Save speech to audio file.

        Args:
            text: Text to convert to speech
            filename: Output filename
            language: Language code ('ml' for Malayalam)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use gTTS for file saving
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filename)
            logger.info(f"Speech saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving speech to file: {e}")
            return False

    def test_voice(self, test_text: str = "ഇത് ഒരു പരീക്ഷണ സന്ദേശമാണ്") -> bool:
        """
        Test the TTS engine with sample text.

        Args:
            test_text: Text to use for testing

        Returns:
            True if test successful, False otherwise
        """
        logger.info("Testing TTS engine...")
        return self.speak_text(test_text, blocking=True)

    def get_engine_info(self) -> Dict[str, str]:
        """
        Get information about the current TTS engine.

        Returns:
            Dictionary with engine information
        """
        info = {
            "engine_type": self.engine_type,
            "is_speaking": str(self.is_speaking),
            "pygame_available": str(self.pygame_available),
        }

        if self.pyttsx3_engine:
            try:
                current_voice = self.pyttsx3_engine.getProperty("voice")
                info["current_voice"] = current_voice if current_voice else "default"
                info["rate"] = str(self.pyttsx3_engine.getProperty("rate"))
                info["volume"] = str(self.pyttsx3_engine.getProperty("volume"))
            except Exception as e:
                info["pyttsx3_error"] = str(e)

        return info

    def __del__(self):
        """Cleanup resources."""
        try:
            self.stop_speaking()
            if self.pyttsx3_engine:
                self.pyttsx3_engine.stop()
        except:
            pass
