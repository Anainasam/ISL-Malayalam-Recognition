#!/usr/bin/env python3
"""
🎯 40 Contextual Words ISL Training
=================================
Focused training on the 40 most contextual and frequently used words
to achieve higher accuracy through better data distribution.
"""

import os
import logging
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import random

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextualISLTrainer:
    def __init__(self):
        self.base_path = Path("datasets/INCLUDE/videos/extracted")
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Select 40 most contextual words - focusing on practical daily use
        self.contextual_words = [
            # Essential Greetings (8 words) - High frequency, clear gestures
            "Hello", "Good Morning", "Good afternoon", "Good evening", 
            "Good night", "Thank you", "Alright", "How are you",
            
            # Colors (8 words) - Most distinguishable colors
            "Red", "Blue", "Green", "Yellow", "Black", "White", "Brown", "Pink",
            
            # Family & People (10 words) - Essential relationships
            "Mother", "Father", "Brother", "Sister", "Son", "Daughter", 
            "Baby", "Man", "Woman", "Parent",
            
            # Home & Objects (8 words) - Daily use items
            "Table", "Chair", "Bed", "Door", "Window", "Kitchen", "Bathroom", "Book",
            
            # Days (6 words) - Most commonly used days
            "Today", "Sunday", "Monday", "Friday", "Saturday", "Wednesday"
        ]
        
        logger.info(f"🎯 Selected {len(self.contextual_words)} contextual words for training")
        
    def get_contextual_folders(self):
        """Get folders for the selected contextual words"""
        folders = []
        word_to_folder = {}
        
        categories = ['Colours', 'Days_and_Time', 'Greetings', 'Home', 'People']
        
        for category in categories:
            category_path = self.base_path / category
            if category_path.exists():
                for folder in category_path.iterdir():
                    if folder.is_dir():
                        folder_name = folder.name
                        parts = folder_name.split('. ', 1)
                        if len(parts) == 2:
                            word = parts[1].strip()
                            if word in self.contextual_words:
                                folders.append(folder)
                                word_to_folder[word] = folder
        
        # Check if we found all contextual words
        found_words = set(word_to_folder.keys())
        missing_words = set(self.contextual_words) - found_words
        if missing_words:
            logger.warning(f"Missing words: {missing_words}")
            # Update contextual_words to only include found words
            self.contextual_words = [w for w in self.contextual_words if w in found_words]
        
        logger.info(f"📂 Found {len(folders)} contextual word folders")
        return folders, word_to_folder

    def extract_enhanced_features(self, video_path):
        """Extract comprehensive video features with better error handling"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            features = []
            frame_count = 0
            max_frames = 50
            
            # Motion and appearance features
            prev_frame = None
            intensities = []
            motion_vectors = []
            
            while cap.read()[0] and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Intensity statistics
                intensities.append([
                    np.mean(gray), np.std(gray), 
                    np.min(gray), np.max(gray)
                ])
                
                # Motion features
                if prev_frame is not None:
                    # Optical flow approximation
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_vectors.append([
                        np.mean(diff), np.std(diff),
                        np.percentile(diff, 90)
                    ])
                
                prev_frame = gray
            
            cap.release()
            
            if len(intensities) == 0:
                return None
            
            # Compile comprehensive features
            intensity_features = np.array(intensities)
            motion_features = np.array(motion_vectors) if motion_vectors else np.zeros((len(intensities)-1, 3))
            
            # Statistical features
            video_features = []
            
            # Intensity statistics across all frames
            video_features.extend([
                np.mean(intensity_features[:, 0]),  # Mean intensity
                np.std(intensity_features[:, 0]),   # Std intensity
                np.mean(intensity_features[:, 1]),  # Mean std
                np.std(intensity_features[:, 1]),   # Std of std
                np.mean(intensity_features[:, 2]),  # Mean min
                np.mean(intensity_features[:, 3]),  # Mean max
            ])
            
            # Motion statistics
            if len(motion_features) > 0:
                video_features.extend([
                    np.mean(motion_features[:, 0]),  # Mean motion
                    np.std(motion_features[:, 0]),   # Std motion
                    np.mean(motion_features[:, 1]),  # Mean motion std
                    np.max(motion_features[:, 2]),   # Max motion peak
                ])
            else:
                video_features.extend([0, 0, 0, 0])
            
            # Temporal features
            video_features.extend([
                len(intensities),  # Frame count
                np.ptp(intensity_features[:, 0]),  # Intensity range
            ])
            
            return np.array(video_features)
            
        except Exception as e:
            logger.warning(f"Error processing video {video_path}: {e}")
            return None

    def load_contextual_dataset(self):
        """Load dataset for contextual words with enhanced features"""
        logger.info("📂 Loading contextual dataset...")
        
        X_data = []
        y_data = []
        word_counts = {}
        
        folders, word_to_folder = self.get_contextual_folders()
        
        for folder in folders:
            folder_name = folder.name
            word = folder_name.split('. ', 1)[1].strip()
            
            logger.info(f"Processing {word}...")
            
            # Get video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.MOV']
            video_files = []
            for ext in video_extensions:
                video_files.extend(list(folder.glob(ext)))
            
            word_features = []
            for video_file in video_files:
                features = self.extract_enhanced_features(video_file)
                if features is not None:
                    word_features.append(features)
            
            if len(word_features) > 0:
                word_counts[word] = len(word_features)
                X_data.extend(word_features)
                y_data.extend([word] * len(word_features))
                
                logger.info(f"   ✅ {word}: {len(word_features)} samples")
        
        X_data = np.array(X_data)
        
        logger.info(f"✅ Loaded {len(X_data)} samples from {len(word_counts)} contextual words")
        logger.info(f"📊 Feature dimensions: {X_data.shape}")
        
        return X_data, y_data, word_counts

    def train_contextual_model(self):
        """Train optimized model on contextual words"""
        logger.info("🚀 Training Contextual ISL Model...")
        
        # Load data
        X_data, y_data, word_counts = self.load_contextual_dataset()
        
        if len(X_data) == 0:
            logger.error("No data loaded!")
            return 0.0
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_data)
        
        # Apply PCA for dimensionality reduction
        n_components = min(100, X_data.shape[1], len(X_data) // 2)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_data)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"📊 After PCA: {n_components} features (explained variance: {explained_variance:.2%})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"📊 Training: {len(X_train)}, Testing: {len(X_test)}")
        
        # Hyperparameter tuning for contextual words
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        logger.info("🔧 Hyperparameter tuning for contextual words...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"🏆 Best parameters: {grid_search.best_params_}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Final evaluation
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"📊 Cross-validation: {cv_mean:.3f} ± {cv_std:.3f}")
        logger.info(f"📊 Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"contextual_40words_model_{timestamp}"
        
        # Save model components
        joblib.dump(best_model, self.model_path / f"{model_name}.pkl")
        joblib.dump(label_encoder, self.model_path / f"contextual_40words_labels_{timestamp}.pkl")
        joblib.dump(pca, self.model_path / f"contextual_40words_pca_{timestamp}.pkl")
        
        # Save training info
        training_info = {
            "model_type": "random_forest_contextual_40words",
            "num_classes": len(label_encoder.classes_),
            "classes": label_encoder.classes_.tolist(),
            "test_accuracy": float(test_accuracy),
            "cv_accuracy_mean": float(cv_mean),
            "cv_accuracy_std": float(cv_std),
            "total_samples": len(X_data),
            "features_after_pca": n_components,
            "explained_variance": float(explained_variance),
            "best_params": grid_search.best_params_,
            "word_counts": word_counts,
            "selected_words": self.contextual_words,
            "training_date": datetime.now().isoformat()
        }
        
        with open(self.model_path / f"contextual_40words_info_{timestamp}.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        # Create visualization
        self.create_results_visualization(
            label_encoder.classes_, word_counts, test_accuracy, 
            cv_mean, report, timestamp
        )
        
        print(f"\n🎉 CONTEXTUAL 40-WORDS MODEL RESULTS:")
        print(f"📊 Test Accuracy: {test_accuracy*100:.1f}%")
        print(f"📊 Cross-validation: {cv_mean*100:.1f}% ± {cv_std*100:.1f}%")
        print(f"📊 Total Words: {len(label_encoder.classes_)}")
        print(f"📊 Total Samples: {len(X_data)}")
        print(f"💾 Model saved as: {model_name}.pkl")
        
        if test_accuracy >= 0.70:
            print("✅ SUCCESS: Achieved 70%+ accuracy target!")
        elif test_accuracy >= 0.60:
            print("🔶 Good progress! Close to 70% target.")
        else:
            print("⚠️  Below target. Consider further optimization.")
        
        return test_accuracy

    def create_results_visualization(self, classes, word_counts, accuracy, cv_mean, report, timestamp):
        """Create comprehensive results visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        accuracies = ['Test Accuracy', 'CV Mean', 'Target']
        values = [accuracy*100, cv_mean*100, 70]
        colors = ['blue', 'green', 'red']
        
        bars = ax1.bar(accuracies, values, color=colors, alpha=0.7)
        ax1.set_title('40 Contextual Words - Accuracy Results', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample distribution by category
        categories = {
            'Greetings': ['Hello', 'Good Morning', 'Good afternoon', 'Good evening', 'Good night', 'Thank you', 'Alright', 'How are you'],
            'Colors': ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White', 'Brown', 'Pink'],
            'Family': ['Mother', 'Father', 'Brother', 'Sister', 'Son', 'Daughter', 'Baby', 'Man', 'Woman', 'Parent'],
            'Home': ['Table', 'Chair', 'Bed', 'Door', 'Window', 'Kitchen', 'Bathroom', 'Book'],
            'Days': ['Today', 'Sunday', 'Monday', 'Friday', 'Saturday', 'Wednesday']
        }
        
        category_counts = {}
        for cat, words in categories.items():
            category_counts[cat] = sum(word_counts.get(word, 0) for word in words if word in word_counts)
        
        ax2.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Sample Distribution by Category', fontsize=14, fontweight='bold')
        
        # 3. Word frequency distribution
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        words, counts = zip(*sorted_words)
        
        ax3.barh(range(len(words)), counts)
        ax3.set_yticks(range(len(words)))
        ax3.set_yticklabels(words, fontsize=8)
        ax3.set_xlabel('Number of Samples')
        ax3.set_title('Samples per Word', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics
        if 'weighted avg' in report:
            metrics = ['precision', 'recall', 'f1-score']
            scores = [report['weighted avg'][metric] for metric in metrics]
            
            bars = ax4.bar(metrics, scores, color=['orange', 'purple', 'brown'], alpha=0.7)
            ax4.set_title('Performance Metrics (Weighted Average)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Score')
            ax4.set_ylim(0, 1)
            
            for bar, score in zip(bars, scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.model_path / f"contextual_40words_results_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Results visualization saved")

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🎯 ISL 40 CONTEXTUAL WORDS TRAINING")
    print("="*70)
    print("📊 Selected words: Most practical and distinguishable")
    print("🎯 Target accuracy: 70%+")
    print("🧠 Method: Optimized Random Forest with Enhanced Features")
    print("="*70)
    
    trainer = ContextualISLTrainer()
    accuracy = trainer.train_contextual_model()
    
    return accuracy

if __name__ == "__main__":
    main()
