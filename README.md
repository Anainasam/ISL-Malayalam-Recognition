# ISL-Malayalam Translation System

A comprehensive system for translating Indian Sign Language (ISL) to Malayalam text and speech using computer vision, machine learning, and natural language processing.

## Features

### 🎯 Core Functionality
- **Real-time Sign Language Recognition**: Uses MediaPipe and OpenCV for hand landmark detection
- **Malayalam Text Processing**: Advanced text processing with grammar correction and language utilities
- **Speech Synthesis**: Converts Malayalam text to speech using multiple TTS engines
- **Modern GUI**: User-friendly interface built with CustomTkinter

### 🔧 Technical Capabilities
- **Computer Vision**: Hand gesture recognition and tracking
- **Machine Learning**: Sign classification using TensorFlow
- **Text Processing**: Malayalam language support with Unicode handling
- **Multi-modal Output**: Both text and audio output for translations
- **Real-time Processing**: Live camera feed processing for instant translation

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera for sign language input
- Windows/Linux/macOS support

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ISL-Malayalam
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

### Starting the Application
1. Launch the application using `python main.py`
2. Click "Start Camera" to begin video capture
3. Position your hands in the camera view
4. Click "Start Recording" to begin sign recognition
5. Perform ISL gestures in front of the camera
6. View translated Malayalam text in the interface
7. Use "Speak" button to hear the translation

### GUI Controls
- **Start/Stop Camera**: Toggle camera feed
- **Start/Stop Recording**: Toggle sign recognition
- **Speak**: Convert current text to speech
- **Clear**: Clear current recognition text
- **Save**: Save translation to file
- **Settings**: Configure camera and TTS engine

## Project Structure

```
ISL-Malayalam/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── .github/
│   └── copilot-instructions.md     # Copilot development guidelines
├── src/
│   ├── __init__.py
│   ├── sign_recognition/
│   │   ├── __init__.py
│   │   └── detector.py             # Sign language detection
│   ├── text_processing/
│   │   ├── __init__.py
│   │   └── malayalam_processor.py  # Malayalam text processing
│   ├── speech_synthesis/
│   │   ├── __init__.py
│   │   └── tts_engine.py          # Text-to-speech engine
│   └── gui/
│       ├── __init__.py
│       └── main_window.py         # Main GUI application
├── models/                        # ML models (to be added)
├── data/                         # Training data (to be added)
├── logs/                         # Application logs
└── tests/                        # Unit tests (to be added)
```

## Key Components

### Sign Recognition (`src/sign_recognition/detector.py`)
- **MediaPipe Integration**: Hand landmark detection and tracking
- **Feature Extraction**: Convert hand positions to ML features
- **Sign Classification**: ML model for recognizing ISL gestures
- **Real-time Processing**: Efficient video frame processing

### Malayalam Processing (`src/text_processing/malayalam_processor.py`)
- **Unicode Support**: Proper Malayalam script handling
- **Text Correction**: Basic spelling and grammar correction
- **Transliteration**: Support for English to Malayalam conversion
- **Language Analysis**: Sentiment analysis and word frequency

### Speech Synthesis (`src/speech_synthesis/tts_engine.py`)
- **Multiple Engines**: Support for pyttsx3 and gTTS
- **Malayalam Speech**: Proper pronunciation handling
- **Audio Output**: Real-time speech and file generation
- **Voice Customization**: Rate, volume, and voice selection

### GUI Application (`src/gui/main_window.py`)
- **Modern Interface**: CustomTkinter-based UI
- **Real-time Display**: Live camera feed and recognition results
- **Interactive Controls**: Easy-to-use buttons and settings
- **Translation History**: Track and save translation sessions

## Development

### Setting up Development Environment
1. Follow installation instructions above
2. Install development dependencies:
   ```bash
   pip install pytest black flake8 mypy
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

### Contributing Guidelines
- Follow PEP 8 style guidelines
- Use type hints for better code documentation
- Add proper logging for debugging
- Handle errors gracefully
- Test with different cameras and lighting conditions
- Consider Malayalam language nuances

### Training Custom Models
The system supports custom ML models for sign recognition:
1. Prepare training data with ISL gesture videos
2. Use MediaPipe to extract hand landmarks
3. Train classification models using the provided feature extraction
4. Replace the model in `src/sign_recognition/detector.py`

## System Requirements

### Hardware
- **Camera**: USB webcam or built-in camera
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: Multi-core processor for real-time processing
- **Storage**: 1GB free space for dependencies

### Software
- **Python**: 3.8+ with pip
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Graphics**: OpenGL support for GUI rendering

## Troubleshooting

### Common Issues
1. **Camera not working**: Check camera permissions and device index
2. **TTS not speaking**: Verify audio drivers and TTS engine installation
3. **Import errors**: Ensure all dependencies are installed in virtual environment
4. **Performance issues**: Close other applications and check CPU usage

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python main.py
```

## Future Enhancements

### Planned Features
- [ ] Advanced ML models for better sign recognition
- [ ] Support for more ISL vocabulary
- [ ] Sentence construction and grammar correction
- [ ] Multi-user support and user profiles
- [ ] Cloud-based model training and updates
- [ ] Mobile app version
- [ ] Integration with educational platforms

### Research Areas
- Deep learning models for sign language recognition
- Natural language processing for Malayalam
- Real-time sign language translation
- Accessibility features for hearing-impaired users

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- MediaPipe team for hand tracking technology
- OpenCV community for computer vision tools
- Contributors to Malayalam language processing tools
- Indian Sign Language research community

## Support

For support, bug reports, or feature requests:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Provide system information and error logs
4. Include steps to reproduce the problem

---

**Note**: This is an educational and research project. For production use, additional testing and validation are recommended.
