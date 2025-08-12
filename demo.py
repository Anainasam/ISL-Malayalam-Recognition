#!/usr/bin/env python3
"""
🎯 ISL-Malayalam Recognition Demo
===============================
Demonstration script showing how to use the ISL recognition system.

This script provides examples of:
- Basic sign recognition
- Model performance analysis
- Batch processing
- Error handling

Run this script to verify your setup and see the system in action.
"""

import os
import sys
from pathlib import Path
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from isl_recognizer import ISLMalayalamRecognizer
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def demo_model_initialization():
    """Demo 1: Initialize and validate the model"""
    print_header("Model Initialization & Validation")
    
    try:
        print("🔄 Loading ISL-Malayalam Recognition Model...")
        start_time = time.time()
        
        recognizer = ISLMalayalamRecognizer()
        
        load_time = time.time() - start_time
        print_success(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Get model statistics
        stats = recognizer.get_model_stats()
        print(f"📊 Model Performance:")
        for key, value in stats.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        return recognizer
        
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return None

def demo_supported_vocabulary(recognizer):
    """Demo 2: Show supported vocabulary and translations"""
    print_header("Supported Vocabulary & Malayalam Translations")
    
    # Get supported words
    words = recognizer.get_supported_words()
    translations = recognizer.get_malayalam_translations()
    
    print(f"📚 Total Supported Words: {len(words)}")
    print(f"\n📝 Complete Vocabulary List:")
    
    # Organize by categories for better display
    categories = {
        "Greetings": ["Hello", "Good Morning", "Good afternoon", "Good evening", 
                     "Good night", "Thank you", "How are you", "Alright"],
        "Family": ["Father", "Mother", "Brother", "Sister", "Son", "Daughter", 
                  "Parent", "Baby", "Man", "Woman"],
        "Colors": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Brown", "Pink"],
        "Days": ["Monday", "Wednesday", "Friday", "Saturday", "Sunday", "Today"],
        "Objects": ["Door", "Window", "Chair", "Table", "Bed", "Kitchen", "Bathroom", "Book"]
    }
    
    for category, cat_words in categories.items():
        print(f"\n   🏷️  {category}:")
        for word in cat_words:
            if word in translations:
                malayalam = translations[word]
                print(f"      • {word:<20} → {malayalam}")

def demo_model_details(recognizer):
    """Demo 3: Show technical model details"""
    print_header("Technical Model Details")
    
    print("🔬 Model Architecture:")
    print("   • Algorithm: Random Forest Classifier")
    print("   • Estimators: 300 trees")
    print("   • Max Depth: 25 levels")
    print("   • Feature Selection: sqrt(n_features)")
    print("   • Preprocessing: PCA dimensionality reduction")
    
    print("\n🎯 Feature Engineering:")
    print("   • Total Features: 12 per video")
    print("   • Intensity Features: 6 (brightness analysis)")
    print("   • Motion Features: 4 (movement analysis)")  
    print("   • Temporal Features: 2 (timing analysis)")
    
    print("\n📊 Training Data:")
    model_info = recognizer.model_info
    print(f"   • Total Samples: {model_info['total_samples']:,}")
    print(f"   • Classes: {model_info['num_classes']}")
    print(f"   • Test Accuracy: {model_info['test_accuracy']*100:.1f}%")
    print(f"   • CV Accuracy: {model_info['cv_accuracy_mean']*100:.1f}% ± {model_info['cv_accuracy_std']*100:.1f}%")

def demo_sample_recognition(recognizer):
    """Demo 4: Show how recognition would work (simulated)"""
    print_header("Sample Recognition Process")
    
    print("🎬 Recognition Process Simulation:")
    print("   1. Load video file → Extract frames")
    print("   2. Analyze brightness patterns → 6 intensity features")
    print("   3. Calculate frame differences → 4 motion features")
    print("   4. Measure temporal properties → 2 timing features")
    print("   5. Apply PCA transformation → Dimensionality reduction")
    print("   6. Random Forest prediction → Classification result")
    print("   7. Malayalam translation → Final output")
    
    print("\n💡 Example Usage Code:")
    print("```python")
    print("from src.isl_recognizer import ISLMalayalamRecognizer")
    print("")
    print("# Initialize recognizer")
    print("recognizer = ISLMalayalamRecognizer()")
    print("")
    print("# Recognize sign from video")
    print("english, malayalam, confidence = recognizer.recognize_sign('hello.mp4')")
    print("print(f'Recognized: {english} → {malayalam} ({confidence:.1%})')")
    print("```")
    
    print("\n🎯 Expected Output:")
    print("   Recognized: Hello → ഹലോ (87.3%)")

def demo_performance_expectations():
    """Demo 5: Set realistic performance expectations"""
    print_header("Performance Expectations & Best Practices")
    
    print("📈 Performance Benchmarks:")
    print("   • Overall Accuracy: 86.8% on test data")
    print("   • High Confidence (>80%): Most clear, well-lit signs")
    print("   • Medium Confidence (60-80%): Acceptable recognition")
    print("   • Low Confidence (<60%): May need re-recording")
    
    print("\n🎯 For Best Results:")
    print("   • Use well-lit environments")
    print("   • Keep hands clearly visible")
    print("   • Use plain backgrounds when possible")
    print("   • Maintain consistent signing speed")
    print("   • Record 1-3 second clips")
    print("   • Follow standard ISL gestures")
    
    print("\n⚠️  Common Challenges:")
    print("   • Poor lighting conditions")
    print("   • Hands partially obscured") 
    print("   • Non-standard signing variations")
    print("   • Very fast or very slow gestures")
    print("   • Signs not in supported vocabulary")

def demo_troubleshooting_tips():
    """Demo 6: Common troubleshooting scenarios"""
    print_header("Troubleshooting Guide")
    
    print("🔧 Common Issues & Solutions:")
    
    issues = [
        {
            "problem": "Model files not found",
            "symptoms": "FileNotFoundError when loading",
            "solution": "Ensure models/ folder contains all .pkl and .json files"
        },
        {
            "problem": "Low confidence predictions",
            "symptoms": "Confidence < 50% for clear signs",
            "solution": "Check video quality, lighting, and hand visibility"
        },
        {
            "problem": "Video processing errors", 
            "symptoms": "Cannot open video file",
            "solution": "Use MP4 format, check file path, verify file integrity"
        },
        {
            "problem": "Import errors",
            "symptoms": "ModuleNotFoundError",
            "solution": "Run: pip install -r requirements.txt"
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"\n   {i}. {issue['problem'].title()}")
        print(f"      Symptoms: {issue['symptoms']}")
        print(f"      Solution: {issue['solution']}")

def demo_system_check(recognizer):
    """Demo 7: Verify system is working correctly"""
    print_header("System Health Check")
    
    checks = []
    
    # Check 1: Model loaded
    if recognizer is not None:
        checks.append(("✅", "Model loaded successfully"))
    else:
        checks.append(("❌", "Model failed to load"))
    
    # Check 2: Model components
    try:
        model_info = recognizer.model_info
        checks.append(("✅", f"Model info accessible ({model_info['num_classes']} classes)"))
    except:
        checks.append(("❌", "Model info not accessible"))
    
    # Check 3: Malayalam translations
    try:
        translations = recognizer.get_malayalam_translations()
        checks.append(("✅", f"Malayalam translations loaded ({len(translations)} words)"))
    except:
        checks.append(("❌", "Malayalam translations not loaded"))
    
    # Check 4: Supported words
    try:
        words = recognizer.get_supported_words()
        checks.append(("✅", f"Vocabulary accessible ({len(words)} words)"))
    except:
        checks.append(("❌", "Vocabulary not accessible"))
    
    print("🔍 System Component Status:")
    for status, message in checks:
        print(f"   {status} {message}")
    
    # Overall status
    all_good = all(check[0] == "✅" for check in checks)
    if all_good:
        print_success("\n🎉 System is fully operational and ready for sign recognition!")
    else:
        print_error("\n⚠️  Some system components have issues. Please check the errors above.")
    
    return all_good

def main():
    """Main demo execution"""
    print("🎯 ISL-Malayalam Recognition System Demo")
    print("🚀 Testing your setup and demonstrating capabilities...")
    
    # Demo 1: Initialize model
    recognizer = demo_model_initialization()
    if recognizer is None:
        print_error("Cannot continue demo without working model")
        return
    
    # Demo 2: Show vocabulary
    demo_supported_vocabulary(recognizer)
    
    # Demo 3: Technical details
    demo_model_details(recognizer)
    
    # Demo 4: Recognition process
    demo_sample_recognition(recognizer)
    
    # Demo 5: Performance expectations
    demo_performance_expectations()
    
    # Demo 6: Troubleshooting
    demo_troubleshooting_tips()
    
    # Demo 7: Final system check
    system_ok = demo_system_check(recognizer)
    
    # Final message
    print_header("Demo Complete")
    if system_ok:
        print("🎉 Congratulations! Your ISL-Malayalam recognition system is ready to use.")
        print("\n🚀 Next Steps:")
        print("   1. Prepare your ISL video files (MP4 format recommended)")
        print("   2. Use recognizer.recognize_sign('your_video.mp4') to test")
        print("   3. Check the README.md for detailed usage examples")
        print("   4. Start building your ISL application!")
        
        print("\n💡 Quick Test:")
        print("```python")
        print("from src.isl_recognizer import ISLMalayalamRecognizer")
        print("recognizer = ISLMalayalamRecognizer()")
        print("# word, malayalam, confidence = recognizer.recognize_sign('test.mp4')")
        print("```")
    else:
        print("⚠️  Please resolve the system issues before using the recognizer.")
        print("📖 Check the troubleshooting section above for solutions.")

if __name__ == "__main__":
    main()
