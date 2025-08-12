#!/usr/bin/env python3
"""
🚀 ISL-Malayalam Setup Verification
==================================
Quick setup check for your production model.

Run this script to verify everything is working correctly.
"""

import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing.append(pip_name)
                print(f"❌ {pip_name} - Not installed")
            else:
                print(f"✅ {pip_name} - Installed")
        except ImportError:
            missing.append(pip_name)
            print(f"❌ {pip_name} - Import error")
    
    return missing

def check_model_files():
    """Check if model files exist"""
    model_dir = Path("models")
    timestamp = "20250810_185558"
    
    required_files = [
        f"contextual_40words_model_{timestamp}.pkl",
        f"contextual_40words_labels_{timestamp}.pkl", 
        f"contextual_40words_pca_{timestamp}.pkl",
        f"contextual_40words_info_{timestamp}.json"
    ]
    
    if not model_dir.exists():
        print(f"❌ Models directory not found: {model_dir}")
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} - Found")
        else:
            missing_files.append(file_name)
            print(f"❌ {file_name} - Missing")
    
    return len(missing_files) == 0

def check_recognizer():
    """Test if the recognizer can be imported and initialized"""
    try:
        sys.path.append(str(Path("src")))
        from isl_recognizer import ISLMalayalamRecognizer
        
        print("✅ ISL recognizer module - Imported successfully")
        
        # Try to initialize
        recognizer = ISLMalayalamRecognizer()
        print("✅ ISL recognizer - Initialized successfully")
        
        # Check basic functionality
        words = recognizer.get_supported_words()
        print(f"✅ Vocabulary - {len(words)} words loaded")
        
        stats = recognizer.get_model_stats()
        print(f"✅ Model stats - {stats['accuracy']} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ ISL recognizer - Error: {e}")
        return False

def main():
    """Run all setup checks"""
    print("🚀 ISL-Malayalam Production Model Setup Check")
    print("=" * 50)
    
    all_good = True
    
    # Check 1: Python version
    print("\n1. Python Version Check:")
    if not check_python_version():
        all_good = False
    
    # Check 2: Dependencies
    print("\n2. Dependencies Check:")
    missing_deps = check_dependencies()
    if missing_deps:
        all_good = False
        print(f"\n💡 Install missing packages: pip install {' '.join(missing_deps)}")
    
    # Check 3: Model files
    print("\n3. Model Files Check:")
    if not check_model_files():
        all_good = False
        print("\n💡 Ensure all model files are in the models/ directory")
    
    # Check 4: Recognizer functionality  
    print("\n4. Recognizer Functionality Check:")
    if not check_recognizer():
        all_good = False
    
    # Final result
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 SUCCESS: Your ISL-Malayalam system is ready to use!")
        print("\n🚀 Next steps:")
        print("   • Run: python demo.py")
        print("   • Read: README.md for detailed usage")
        print("   • Test: Use recognizer.recognize_sign('video.mp4')")
    else:
        print("⚠️  ISSUES FOUND: Please resolve the problems above")
        print("\n🔧 Common solutions:")
        print("   • Install dependencies: pip install -r requirements.txt") 
        print("   • Check model files in models/ directory")
        print("   • Verify Python 3.8+ installation")

if __name__ == "__main__":
    main()
