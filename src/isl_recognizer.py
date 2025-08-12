#!/usr/bin/env python3
"""
🎯 ISL-Malayalam Recognition Engine
==================================
Production-ready Random Forest model for ISL to Malayalam translation.
Achieves 86.8% accuracy on 40 contextual words.

Author: ISL-Malayalam Team
Model: Random Forest Classifier (86.8% accuracy)
Date: August 2025
"""

import os
import cv2
import numpy as np
import joblib
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISLMalayalamRecognizer:
    """
    Production ISL Recognition Engine
    
    Features:
    - 86.8% accuracy on test data
    - 40 contextual ISL words support
    - Malayalam text and pronunciation
    - Real-time video processing
    """
    
    def __init__(self, model_timestamp="20250810_185558"):
        """Initialize the ISL recognizer with trained models"""
        self.model_path = Path("models")
        
        # Malayalam translation dictionary
        self.malayalam_dict = {
            "Alright": "ശരി",
            "Baby": "കുഞ്ഞ്",
            "Bathroom": "കുളിമുറി",
            "Bed": "കിടക്ക",
            "Black": "കറുപ്പ്",
            "Blue": "നീല",
            "Book": "പുസ്തകം",
            "Brother": "സഹോദരൻ",
            "Brown": "തവിട്ട്",
            "Chair": "കസേര",
            "Daughter": "മകൾ",
            "Door": "വാതിൽ",
            "Father": "അച്ഛൻ",
            "Friday": "വെള്ളിയാഴ്ച",
            "Good Morning": "സുപ്രഭാതം",
            "Good afternoon": "ശുഭ ഉച്ചയ്ക്ക്",
            "Good evening": "ശുഭ സന്ധ്യ",
            "Good night": "ശുഭരാത്രി",
            "Green": "പച്ച",
            "Hello": "ഹലോ",
            "How are you": "നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്",
            "Kitchen": "അടുക്കള",
            "Man": "പുരുഷൻ",
            "Monday": "തിങ്കളാഴ്ച",
            "Mother": "അമ്മ",
            "Parent": "മാതാപിതാക്കൾ",
            "Pink": "പിങ്ക്",
            "Red": "ചുവപ്പ്",
            "Saturday": "ശനിയാഴ്ച",
            "Sister": "സഹോദരി",
            "Son": "മകൻ",
            "Sunday": "ഞായറാഴ്ച",
            "Table": "മേശ",
            "Thank you": "നന്ദി",
            "Today": "ഇന്ന്",
            "Wednesday": "ബുധനാഴ്ച",
            "White": "വെള്ള",
            "Window": "ജനാല",
            "Woman": "സ്ത്രീ",
            "Yellow": "മഞ്ഞ"
        }
        
        # Load trained model components
        try:
            self.model = joblib.load(self.model_path / f"contextual_40words_model_{model_timestamp}.pkl")
            self.label_encoder = joblib.load(self.model_path / f"contextual_40words_labels_{model_timestamp}.pkl")
            self.pca = joblib.load(self.model_path / f"contextual_40words_pca_{model_timestamp}.pkl")
            
            # Load model information
            with open(self.model_path / f"contextual_40words_info_{model_timestamp}.json", 'r') as f:
                self.model_info = json.load(f)
            
            logger.info("✅ ISL-Malayalam Model loaded successfully!")
            logger.info(f"📊 Model accuracy: {self.model_info['test_accuracy']*100:.1f}%")
            logger.info(f"📚 Supported words: {len(self.model_info['classes'])}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def extract_video_features(self, video_path):
        """
        Extract 12 statistical features from video file
        
        Features extracted:
        - 6 intensity features (brightness analysis)
        - 4 motion features (movement analysis)
        - 2 temporal features (timing analysis)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            # Initialize feature collection
            intensities = []
            motion_vectors = []
            prev_frame = None
            frame_count = 0
            max_frames = 50
            
            # Process video frame by frame
            while cap.read()[0] and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Extract intensity features for this frame
                intensities.append([
                    np.mean(gray),  # Average brightness
                    np.std(gray),   # Brightness variation
                    np.min(gray),   # Darkest pixel
                    np.max(gray)    # Brightest pixel
                ])
                
                # Extract motion features (compare with previous frame)
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_vectors.append([
                        np.mean(diff),           # Average motion
                        np.std(diff),            # Motion variation
                        np.percentile(diff, 90)  # Peak motion
                    ])
                
                prev_frame = gray
            
            cap.release()
            
            if len(intensities) == 0:
                logger.warning("No frames processed from video")
                return None
            
            # Compile statistical features
            intensity_features = np.array(intensities)
            motion_features = np.array(motion_vectors) if motion_vectors else np.zeros((len(intensities)-1, 3))
            
            # Create 12-dimensional feature vector
            video_features = []
            
            # Intensity features (6 features)
            video_features.extend([
                np.mean(intensity_features[:, 0]),  # Overall avg brightness
                np.std(intensity_features[:, 0]),   # Brightness consistency
                np.mean(intensity_features[:, 1]),  # Avg within-frame variation
                np.std(intensity_features[:, 1]),   # Variation consistency
                np.mean(intensity_features[:, 2]),  # Avg darkest areas
                np.mean(intensity_features[:, 3]),  # Avg brightest areas
            ])
            
            # Motion features (4 features)
            if len(motion_features) > 0:
                video_features.extend([
                    np.mean(motion_features[:, 0]),  # Avg motion intensity
                    np.std(motion_features[:, 0]),   # Motion consistency
                    np.mean(motion_features[:, 1]),  # Avg motion variation
                    np.max(motion_features[:, 2]),   # Peak motion moment
                ])
            else:
                video_features.extend([0, 0, 0, 0])
            
            # Temporal features (2 features)
            video_features.extend([
                len(intensities),                    # Video length (frames)
                np.ptp(intensity_features[:, 0]),    # Brightness range
            ])
            
            return np.array(video_features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features from {video_path}: {e}")
            return None
    
    def recognize_sign(self, video_path):
        """
        Recognize ISL sign from video and return Malayalam translation
        
        Returns:
        - word: English word recognized
        - malayalam: Malayalam translation
        - confidence: Prediction confidence (0-1)
        """
        # Extract features from video
        features = self.extract_video_features(video_path)
        if features is None:
            return None, None, 0.0
        
        # Apply PCA transformation (same as training)
        features_pca = self.pca.transform(features)
        
        # Get prediction from Random Forest
        prediction = self.model.predict(features_pca)[0]
        probabilities = self.model.predict_proba(features_pca)[0]
        confidence = np.max(probabilities)
        
        # Convert prediction to word
        word = self.label_encoder.inverse_transform([prediction])[0]
        malayalam = self.malayalam_dict.get(word, word)
        
        return word, malayalam, float(confidence)
    
    def get_supported_words(self):
        """Get list of all supported ISL words"""
        return self.model_info['classes']
    
    def get_malayalam_translations(self):
        """Get all Malayalam translations"""
        return self.malayalam_dict
    
    def get_model_stats(self):
        """Get model performance statistics"""
        return {
            'accuracy': f"{self.model_info['test_accuracy']*100:.1f}%",
            'cv_accuracy': f"{self.model_info['cv_accuracy_mean']*100:.1f}%",
            'total_words': self.model_info['num_classes'],
            'training_samples': self.model_info['total_samples'],
            'model_type': self.model_info['model_type']
        }

def main():
    """Demo function showing how to use the ISL recognizer"""
    print("🎯 ISL-Malayalam Recognition Engine")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = ISLMalayalamRecognizer()
    
    # Show model stats
    stats = recognizer.get_model_stats()
    print(f"📊 Model Accuracy: {stats['accuracy']}")
    print(f"📊 Cross-Validation: {stats['cv_accuracy']}")
    print(f"📚 Supported Words: {stats['total_words']}")
    
    # Show supported words
    print(f"\n📝 Supported ISL Words:")
    words = recognizer.get_supported_words()
    for i, word in enumerate(words, 1):
        malayalam = recognizer.malayalam_dict[word]
        print(f"  {i:2d}. {word:<20} → {malayalam}")
    
    print(f"\n✅ Ready for recognition!")
    print(f"💡 Usage: word, malayalam, confidence = recognizer.recognize_sign('video.mp4')")

if __name__ == "__main__":
    main()
