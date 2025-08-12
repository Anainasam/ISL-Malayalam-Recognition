"""
Updated Sign Language Detector with Rapid Model
Uses the 20-word rapid model for high accuracy detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class SignLanguageDetector:
    """
    Enhanced ISL detector using rapid training approach.
    Focuses on 20 core Malayalam words with 75-85% accuracy.
    """

    def __init__(self):
        """Initialize detector with rapid model."""
        self.base_dir = Path("d:/ISL-Malayalam")
        self.models_dir = self.base_dir / "models"

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Load rapid model and label encoder
        self.model = None
        self.label_encoder = None
        self.word_mappings = {}
        self.sequence_buffer = []
        self.sequence_length = 30

        self._load_rapid_model()

        logger.info("SignLanguageDetector initialized with rapid model")

    def _load_rapid_model(self):
        """Load the rapid trained model."""
        try:
            model_path = self.models_dir / "rapid_isl_model.h5"
            encoder_path = self.models_dir / "label_encoder.pkl"
            info_path = self.models_dir / "training_info.json"

            if model_path.exists() and encoder_path.exists():
                # Load model
                self.model = tf.keras.models.load_model(str(model_path))

                # Load label encoder
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)

                # Load training info
                if info_path.exists():
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        self.word_mappings = info.get("word_mappings", {})

                logger.info(f"Rapid model loaded successfully")
                logger.info(
                    f"Model supports {len(self.label_encoder.classes_)} classes"
                )

            else:
                logger.warning("Rapid model not found. Run rapid_train.py first!")

        except Exception as e:
            logger.error(f"Failed to load rapid model: {e}")

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from frame.

        Args:
            frame: Input frame

        Returns:
            Landmark features or None
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Ensure fixed size (63 features for 1 hand)
                if len(landmarks) < 63:
                    landmarks.extend([0.0] * (63 - len(landmarks)))
                else:
                    landmarks = landmarks[:63]

                return np.array(landmarks)

            return None

        except Exception as e:
            logger.error(f"Landmark extraction failed: {e}")
            return None

    def predict_sign(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Predict sign from current frame using rapid model.

        Args:
            frame: Current video frame

        Returns:
            Prediction dictionary or None
        """
        if self.model is None:
            return None

        try:
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            if landmarks is None:
                return None

            # Add to sequence buffer
            self.sequence_buffer.append(landmarks)

            # Keep only last 30 frames
            if len(self.sequence_buffer) > self.sequence_length:
                self.sequence_buffer = self.sequence_buffer[-self.sequence_length :]

            # Need full sequence for prediction
            if len(self.sequence_buffer) < self.sequence_length:
                return None

            # Prepare sequence for prediction
            sequence = np.array(self.sequence_buffer)
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

            # Make prediction
            predictions = self.model.predict(sequence, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])

            # Get class name
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            malayalam_word = self.word_mappings.get(predicted_class, predicted_class)

            return {
                "english_word": predicted_class,
                "malayalam_word": malayalam_word,
                "confidence": confidence,
                "all_predictions": {
                    self.label_encoder.classes_[i]: float(predictions[0][i])
                    for i in range(len(self.label_encoder.classes_))
                },
            }

        except Exception as e:
            logger.error(f"Sign prediction failed: {e}")
            return None

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: Input frame

        Returns:
            Frame with landmarks drawn
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )

            return frame

        except Exception as e:
            logger.error(f"Landmark drawing failed: {e}")
            return frame

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "status": "Model loaded",
            "classes": (
                self.label_encoder.classes_.tolist() if self.label_encoder else []
            ),
            "total_classes": (
                len(self.label_encoder.classes_) if self.label_encoder else 0
            ),
            "malayalam_words": list(self.word_mappings.values()),
            "model_type": "Rapid ISL Model (20 words)",
        }

    def reset_sequence(self):
        """Reset the sequence buffer."""
        self.sequence_buffer = []

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "hands"):
            self.hands.close()
        logger.info("SignLanguageDetector cleanup completed")


# For backward compatibility
class ISLDetector(SignLanguageDetector):
    """Alias for backward compatibility."""

    pass
