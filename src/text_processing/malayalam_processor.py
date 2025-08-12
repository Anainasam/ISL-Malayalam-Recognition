"""
Malayalam Text Processing Module
Handles Malayalam text processing, grammar correction, and language utilities.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import nltk
from indic_transliteration import sanscript

logger = logging.getLogger(__name__)


class MalayalamProcessor:
    """
    Malayalam text processing and language utilities.
    Handles text correction, grammar checking, and linguistic processing.
    """

    def __init__(self):
        """Initialize Malayalam processor."""
        self.malayalam_chars = set(range(0x0D00, 0x0D7F))  # Malayalam Unicode range
        self.common_words = self._load_common_words()
        self.grammar_rules = self._load_grammar_rules()

        # Download required NLTK data
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")

        logger.info("MalayalamProcessor initialized successfully")

    def _load_common_words(self) -> Dict[str, int]:
        """Load common Malayalam words dictionary with ISL-focused vocabulary."""
        # Expanded Malayalam vocabulary based on INCLUDE dataset categories
        # Categories: Greetings, Family, Daily Activities, Objects, Colors, etc.
        common_words = {
            # Greetings and Basic Communication (100-95 priority)
            "നമസ്കാരം": 100,
            "ഹലോ": 100,
            "ധന്യവാദ്": 98,
            "സ്വാഗതം": 95,
            "വിടപറയൽ": 95,
            "ക്ഷമിക്കുക": 97,
            # Family Relations (90-85 priority)
            "അമ്മ": 92,
            "അച്ഛൻ": 92,
            "മകൻ": 88,
            "മകൾ": 88,
            "സഹോദരൻ": 85,
            "സഹോദരി": 85,
            "അമ്മൂമ്മ": 82,
            "അച്ഛമ്മ": 82,
            "ഭർത്താവ്": 80,
            "ഭാര്യ": 80,
            # Daily Objects and Activities (85-75 priority)
            "വെള്ളം": 88,
            "ഭക്ഷണം": 87,
            "വീട്": 86,
            "സ്കൂൾ": 85,
            "പുസ്തകം": 80,
            "പേന": 78,
            "പേപ്പർ": 76,
            "കസേര": 75,
            "മേശ": 75,
            "വാതിൽ": 74,
            "ജാലകം": 73,
            # Colors (80-70 priority)
            "ചുവപ്പ്": 82,
            "നീല": 81,
            "പച്ച": 80,
            "മഞ്ഞ": 79,
            "കറുപ്പ്": 78,
            "വെളുപ്പ്": 77,
            "തവിട്ടുനിറം": 72,
            # Numbers (85-75 priority)
            "ഒന്ന്": 85,
            "രണ്ട്": 84,
            "മൂന്ന്": 83,
            "നാല്": 82,
            "അഞ്ച്": 81,
            "ആറ്": 80,
            "ഏഴ്": 79,
            "എട്ട്": 78,
            "ഒമ്പത്": 77,
            "പത്ത്": 76,
            # Time and Days (75-65 priority)
            "ഇന്ന്": 78,
            "നാളെ": 77,
            "ഇന്നലെ": 76,
            "തിങ്കളാഴ്ച": 72,
            "ചൊവ്വാഴ്ച": 71,
            "ബുധനാഴ്ച": 70,
            "വ്യാഴാഴ്ച": 69,
            "വെള്ളിയാഴ്ച": 68,
            "ശനിയാഴ്ച": 67,
            "ഞായറാഴ്ച": 66,
            "സമയം": 75,
            "മിനിറ്റ്": 70,
            "മണിക്കൂർ": 69,
            # Emotions and Adjectives (80-60 priority)
            "സന്തോഷം": 82,
            "സങ്കടം": 78,
            "സ്നേഹം": 80,
            "ദേഷ്യം": 75,
            "ഭയം": 73,
            "നല്ല": 85,
            "ചീത്ത": 75,
            "വലിയ": 72,
            "ചെറിയ": 70,
            "ഉയരമുള്ള": 68,
            "കുറിയ": 67,
            # Actions and Verbs (85-70 priority)
            "വരിക": 85,
            "പോകുക": 84,
            "കാണുക": 83,
            "കേൾക്കുക": 82,
            "പറയുക": 81,
            "ഇരിക്കുക": 80,
            "നിൽക്കുക": 79,
            "കിടക്കുക": 78,
            "തിന്നുക": 77,
            "കുടിക്കുക": 76,
            "എഴുതുക": 75,
            "വായിക്കുക": 74,
            # Places (75-65 priority)
            "ആശുപത്രി": 75,
            "കടപ്പ്": 74,
            "പള്ളി": 73,
            "ക്ഷേത്രം": 72,
            "മുറി": 71,
            "അടുക്കള": 70,
            "കുളിമുറി": 69,
            "പൂന്തോട്ടം": 68,
            "തെരുവ്": 67,
            # Transportation (70-60 priority)
            "കാർ": 75,
            "ബസ്": 74,
            "ട്രെയിൻ": 73,
            "വിമാനം": 72,
            "സൈക്കിൾ": 71,
            "ബൈക്": 70,
            # Animals (70-60 priority)
            "നായ": 75,
            "പൂച്ച": 74,
            "പശു": 73,
            "കോഴി": 72,
            "പക്ഷി": 71,
            # Common ISL Words from Research
            "സഹായം": 88,
            "പാഠം": 75,
            "ക്ലാസ്": 72,
            "ടീച്ചർ": 78,
            "വിദ്യാർത്ഥി": 80,
            "സുഹൃത്ത്": 83,
            "ആരോഗ്യം": 76,
            "രോഗം": 71,
            "മരുന്ന്": 70,
            "ഡോക്ടർ": 75,
            "നഴ്സ്": 72,
            # Technology and Modern Words (65-55 priority)
            "കമ്പ്യൂട്ടർ": 68,
            "ഫോൺ": 70,
            "ഇന്റർനെറ്റ്": 65,
            "ടെലിവിഷൻ": 67,
            "റേഡിയോ": 60,
            # Weather and Nature (65-55 priority)
            "മഴ": 68,
            "വെയിൽ": 67,
            "കാറ്റ്": 66,
            "തണുപ്പ്": 65,
            "ചൂട്": 64,
            "മേഘം": 63,
            "നക്ഷത്രം": 60,
            "ചന്ദ്രൻ": 62,
            "സൂര്യൻ": 65,
        }
        return common_words

    def _load_grammar_rules(self) -> Dict[str, List[str]]:
        """Load basic Malayalam grammar rules."""
        return {
            "conjunctions": ["ഉം", "ഓ", "എന്നാൽ", "പക്ഷേ", "അല്ലെങ്കിൽ"],
            "question_words": ["എന്ത്", "ഏത്", "എവിടെ", "എപ്പോൾ", "എങ്ങനെ", "എന്തുകൊണ്ട്"],
            "pronouns": ["ഞാൻ", "നീ", "അവൻ", "അവൾ", "നാം", "നിങ്ങൾ", "അവർ"],
            "common_suffixes": ["ൻ്റെ", "യുടെ", "ിൽ", "ിൽനിന്ന്", "ോട്", "ിന്"],
        }

    def is_malayalam_text(self, text: str) -> bool:
        """
        Check if the given text contains Malayalam characters.

        Args:
            text: Input text to check

        Returns:
            True if text contains Malayalam characters, False otherwise
        """
        if not text:
            return False

        malayalam_count = 0
        total_chars = 0

        for char in text:
            if char.isalpha():
                total_chars += 1
                if ord(char) in self.malayalam_chars:
                    malayalam_count += 1

        if total_chars == 0:
            return False

        # Consider it Malayalam if more than 50% chars are Malayalam
        return (malayalam_count / total_chars) > 0.5

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Malayalam text.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove unwanted characters but keep Malayalam punctuation
        text = re.sub(r"[^\u0D00-\u0D7F\s\.\,\!\?\:\;]", "", text)

        return text

    def correct_spelling(self, text: str) -> str:
        """
        Basic spelling correction for Malayalam text.

        Args:
            text: Input text to correct

        Returns:
            Spell-corrected text
        """
        if not text or not self.is_malayalam_text(text):
            return text

        words = text.split()
        corrected_words = []

        for word in words:
            cleaned_word = self.clean_text(word)

            # Check if word exists in common words
            if cleaned_word in self.common_words:
                corrected_words.append(cleaned_word)
            else:
                # Find closest match using simple edit distance
                closest_word = self._find_closest_word(cleaned_word)
                corrected_words.append(closest_word if closest_word else cleaned_word)

        return " ".join(corrected_words)

    def _find_closest_word(self, word: str) -> Optional[str]:
        """
        Find closest matching word in vocabulary.

        Args:
            word: Input word to match

        Returns:
            Closest matching word or None
        """
        if not word:
            return None

        min_distance = float("inf")
        closest_word = None

        for vocab_word in self.common_words:
            distance = self._edit_distance(word, vocab_word)
            if (
                distance < min_distance and distance <= 2
            ):  # Allow max 2 character differences
                min_distance = distance
                closest_word = vocab_word

        return closest_word

    def _edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def add_punctuation(self, text: str) -> str:
        """
        Add appropriate punctuation to Malayalam text.

        Args:
            text: Input text

        Returns:
            Text with added punctuation
        """
        if not text:
            return ""

        # Add period at end if missing
        text = text.strip()
        if text and not text.endswith((".", "!", "?", ":")):
            # Check if it's a question
            for question_word in self.grammar_rules["question_words"]:
                if question_word in text:
                    text += "?"
                    break
            else:
                text += "."

        return text

    def transliterate_to_malayalam(
        self, text: str, source_script: str = "itrans"
    ) -> str:
        """
        Transliterate text to Malayalam from other scripts.

        Args:
            text: Input text to transliterate
            source_script: Source script (default: itrans)

        Returns:
            Transliterated Malayalam text
        """
        try:
            if source_script.lower() == "itrans":
                return sanscript.transliterate(
                    text, sanscript.ITRANS, sanscript.MALAYALAM
                )
            elif source_script.lower() == "english":
                # Simple English to Malayalam transliteration mapping
                return self._english_to_malayalam(text)
            else:
                return text
        except Exception as e:
            logger.error(f"Transliteration error: {e}")
            return text

    def _english_to_malayalam(self, text: str) -> str:
        """
        Simple English to Malayalam word mapping.

        Args:
            text: English text

        Returns:
            Malayalam equivalent
        """
        english_to_malayalam = {
            "hello": "നമസ്കാരം",
            "thank you": "ധന്യവാദ്",
            "thanks": "ധന്യവാദ്",
            "sorry": "ക്ഷമിക്കുക",
            "help": "സഹായം",
            "water": "വെള്ളം",
            "food": "ഭക്ഷണം",
            "home": "വീട്",
            "house": "വീട്",
            "school": "സ്കൂൾ",
            "mother": "അമ്മ",
            "father": "അച്ഛൻ",
            "child": "കുട്ടി",
            "book": "പുസ്തകം",
            "lesson": "പാഠം",
            "class": "ക്ലാസ്",
            "teacher": "ടീച്ചർ",
            "student": "വിദ്യാർത്ഥി",
            "friend": "സുഹൃത്ത്",
            "happy": "സന്തോഷം",
            "sad": "സങ്കടം",
            "love": "സ്നേഹം",
        }

        text_lower = text.lower().strip()
        return english_to_malayalam.get(text_lower, text)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Basic sentiment analysis for Malayalam text.

        Args:
            text: Malayalam text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not text or not self.is_malayalam_text(text):
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        # Simple rule-based sentiment analysis
        positive_words = ["സന്തോഷം", "സ്നേഹം", "നല്ല", "സുന്ദരൻ", "സുന്दരി", "മികച്ച"]
        negative_words = ["സങ്കടം", "ദുഃഖം", "വേദന", "കോപം", "ചീത്ത"]

        words = text.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)

        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score

        return {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": max(0.0, neutral_score),
        }

    def get_word_frequency(self, text: str) -> Dict[str, int]:
        """
        Get word frequency analysis of Malayalam text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with word frequencies
        """
        if not text:
            return {}

        words = self.clean_text(text).split()
        frequency = {}

        for word in words:
            if word and len(word) > 1:  # Ignore single characters
                frequency[word] = frequency.get(word, 0) + 1

        return dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
