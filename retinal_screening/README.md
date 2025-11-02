# Retinal AI Screening Flutter App

<div align="center">
  <img src="https://img.shields.io/badge/Flutter-3.9.2-blue?logo=flutter" alt="Flutter Version">
  <img src="https://img.shields.io/badge/PyTorch-Mobile-orange?logo=pytorch" alt="PyTorch Mobile">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</div>

## 🎯 Overview

A production-ready Flutter mobile application that leverages AI to detect retinal diseases from fundus images. The app integrates a quantized PyTorch deep learning model capable of identifying 45 different retinal conditions with clinical recommendations.

## ✨ Features

### 🔬 AI-Powered Analysis
- **45 Disease Detection**: Comprehensive retinal disease classification
- **Real-time Inference**: ~200ms processing time on mobile devices
- **Confidence Scoring**: Probability scores for each prediction
- **Uncertainty Metrics**: Model confidence indicators

### 📱 Mobile-First Design
- **Image Capture**: Camera integration for instant analysis
- **Gallery Upload**: Select existing retinal images
- **Offline Capable**: On-device inference (no internet required)
- **Material 3 UI**: Modern, accessible interface

### 🏥 Clinical Features
- **Top 5 Predictions**: Ranked by confidence
- **Severity Levels**: High, Moderate, Low, Very Low
- **Clinical Recommendations**: Actionable next steps
- **Uncertainty Analysis**: Entropy-based confidence metrics

## 🚀 Quick Start

### Prerequisites
- Flutter SDK 3.9.2+
- Android Studio (Android development)
- Android device/emulator (minSdk 21+)

### Installation

1. **Install Dependencies**
   ```bash
   flutter pub get
   ```

2. **Verify Assets**
   ```bash
   ls -lh assets/models/best_model_mobile.pth  # ~119 MB
   ls -lh assets/data/disease_names.json       # Disease mappings
   ```

3. **Run on Android**
   ```bash
   flutter run
   ```

## 📦 Project Structure

```
retinal_screening/
├── lib/
│   ├── main.dart                          # App entry point
│   ├── models/disease_prediction.dart     # Data models
│   ├── providers/analysis_provider.dart   # State management
│   ├── services/model_service.dart        # PyTorch integration
│   └── screens/                           # UI screens
├── assets/
│   ├── models/best_model_mobile.pth       # 119 MB PyTorch model
│   └── data/disease_names.json            # 45 disease names
├── android/app/src/main/kotlin/.../MainActivity.kt  # Platform channel
└── FLUTTER_SETUP_GUIDE.md                 # Detailed setup instructions
```

## 🔬 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | SceneGraphTransformer |
| **Input Shape** | [1, 3, 224, 224] |
| **Output Shape** | [1, 45] |
| **Size** | 119 MB (INT8 Quantized) |
| **Inference Time** | ~200ms (CPU) |
| **Diseases** | 45 retinal conditions |

## 📚 Documentation

- **Setup Guide**: `FLUTTER_SETUP_GUIDE.md` - Comprehensive setup instructions
- **Model Metadata**: `../models/model_metadata.json` - Model specifications

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is a screening tool for educational and research purposes only. It is NOT a medical device and should NOT be used as a replacement for professional medical diagnosis.

Always consult qualified healthcare professionals for proper diagnosis and treatment decisions.

## 🔧 Troubleshooting

See `FLUTTER_SETUP_GUIDE.md` for detailed troubleshooting steps.

**Common issues**:
- Model loading errors → Verify `assets/models/best_model_mobile.pth` exists
- Permission errors → Check AndroidManifest.xml permissions
- Out of memory → Use release build: `flutter run --release`

## 📄 License

MIT License

---

**Built with ❤️ using Flutter and PyTorch Mobile**

