# Configuration file for ISL-Malayalam Translation System

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Sign Language Recognition Settings
SIGN_RECOGNITION = {
    "model_path": MODELS_DIR / "isl_recognition_model.h5",
    "confidence_threshold": 0.7,
    "max_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "camera_resolution": (640, 480),
    "fps_target": 30,
}

# Malayalam Text Processing Settings
TEXT_PROCESSING = {
    "unicode_range": (0x0D00, 0x0D7F),  # Malayalam Unicode range
    "max_edit_distance": 2,
    "spell_check_enabled": True,
    "auto_punctuation": True,
}

# Speech Synthesis Settings
SPEECH_SYNTHESIS = {
    "default_engine": "pyttsx3",  # or "gtts"
    "malayalam_language_code": "ml",
    "speech_rate": 150,  # words per minute
    "volume": 0.8,
    "gtts_timeout": 10,
    "audio_format": "mp3",
}

# GUI Settings
GUI_CONFIG = {
    "theme": "dark",
    "color_theme": "blue",
    "window_size": (1200, 800),
    "min_window_size": (800, 600),
    "video_display_size": (480, 360),
    "font_family": "Arial",
    "malayalam_font": "Noto Sans Malayalam",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "app.log",
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Performance Settings
PERFORMANCE = {
    "max_fps": 30,
    "frame_skip": 1,
    "processing_threads": 2,
    "memory_limit_mb": 512,
}

# Default ISL vocabulary
DEFAULT_VOCABULARY = {
    0: "നമസ്കാരം",  # Hello/Namaste
    1: "ധന്യവാദ്",  # Thank you
    2: "ക്ഷമിക്കുക",  # Sorry
    3: "സഹായം",  # Help
    4: "വെള്ളം",  # Water
    5: "ഭക്ഷണം",  # Food
    6: "വീട്",  # Home
    7: "സ്കൂൾ",  # School
    8: "അമ്മ",  # Mother
    9: "അച്ഛൻ",  # Father
    10: "കുട്ടി",  # Child
    11: "പുസ്തകം",  # Book
    12: "പാഠം",  # Lesson
    13: "ടീച്ചർ",  # Teacher
    14: "വിദ്യാർത്ഥി",  # Student
    15: "സുഹൃത്ത്",  # Friend
    16: "സന്തോഷം",  # Happy
    17: "സങ്കടം",  # Sad
    18: "സ്നേഹം",  # Love
    19: "കളി",  # Play
    20: "പണി",  # Work
}
