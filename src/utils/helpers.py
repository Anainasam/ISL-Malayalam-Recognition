"""
Utility functions for the ISL-Malayalam Translation System
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """
    level = getattr(logging, log_level.upper())

    handlers = [logging.StreamHandler()]
    if log_file:
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with JSON data or None if error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to output file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def get_available_cameras() -> List[int]:
    """
    Get list of available camera indices.

    Returns:
        List of camera indices that are available
    """
    available_cameras = []

    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return available_cameras


def validate_camera_index(camera_index: int) -> bool:
    """
    Validate if a camera index is available.

    Args:
        camera_index: Camera index to validate

    Returns:
        True if camera is available, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        is_available = cap.isOpened()
        cap.release()
        return is_available
    except Exception:
        return False


def resize_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: Input frame
        target_size: Target size (width, height)

    Returns:
        Resized frame
    """
    if frame is None:
        return None

    height, width = frame.shape[:2]
    target_width, target_height = target_size

    # Calculate aspect ratio
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Frame is wider, fit to target width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Frame is taller, fit to target height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create black background and center the resized frame
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    result[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_frame
    )

    return result


def create_directory_structure(base_path: str) -> bool:
    """
    Create the necessary directory structure for the application.

    Args:
        base_path: Base path for the application

    Returns:
        True if successful, False otherwise
    """
    directories = ["data", "models", "logs", "temp", "exports", "recordings"]

    try:
        for directory in directories:
            dir_path = Path(base_path) / directory
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("Directory structure created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        return False


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_malayalam_text(text: str) -> bool:
    """
    Validate if text contains Malayalam characters.

    Args:
        text: Text to validate

    Returns:
        True if text contains Malayalam characters, False otherwise
    """
    if not text:
        return False

    malayalam_range = range(0x0D00, 0x0D7F)
    malayalam_count = sum(1 for char in text if ord(char) in malayalam_range)

    return malayalam_count > 0


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.

    Returns:
        Dictionary with system information
    """
    import platform
    import psutil

    try:
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            "cpu_cores": psutil.cpu_count(),
            "opencv_version": (
                cv2.__version__ if "cv2" in globals() else "Not available"
            ),
        }
        return info
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e)}


def clean_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean temporary files older than specified age.

    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours

    Returns:
        Number of files cleaned
    """
    import time

    cleaned_count = 0
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()

    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return 0

        for file_path in temp_path.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1

        logger.info(f"Cleaned {cleaned_count} temporary files")
        return cleaned_count

    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")
        return 0


def export_translation_history(history: List[Dict[str, Any]], filename: str) -> bool:
    """
    Export translation history to file.

    Args:
        history: List of translation records
        filename: Output filename

    Returns:
        True if successful, False otherwise
    """
    try:
        export_data = {
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_translations": len(history),
            "translations": history,
        }

        return save_json_file(export_data, filename)

    except Exception as e:
        logger.error(f"Error exporting translation history: {e}")
        return False
