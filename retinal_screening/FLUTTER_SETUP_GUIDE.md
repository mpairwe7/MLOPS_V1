# Flutter Retinal AI Screening App - Setup Guide

## Overview

This Flutter application integrates your trained PyTorch retinal disease classification model (`best_model_mobile.pth`) for mobile deployment with a professional UI and explainability features.

## Architecture

### Technologies Used
- **Flutter SDK**: ^3.9.2
- **PyTorch Mobile**: Cross-platform ML inference
- **Provider**: State management
- **Image Picker**: Gallery/camera image selection
- **Material 3**: Modern UI design

### Key Components

1. **ModelService** (`lib/services/model_service.dart`)
   - Loads PyTorch model from assets
   - Preprocesses images (224x224, ImageNet normalization)
   - Runs inference via platform channels
   - Returns structured predictions

2. **AnalysisProvider** (`lib/providers/analysis_provider.dart`)
   - Manages app state (image, loading, results, errors)
   - Implements ChangeNotifier pattern
   - Reactive UI updates

3. **HomeScreen** (`lib/screens/home_screen.dart`)
   - Image selection (gallery/camera)
   - Analysis trigger
   - Loading states

4. **ResultsScreen** (`lib/screens/results_screen.dart`)
   - Top 5 disease predictions
   - Confidence visualizations
   - Clinical recommendations
   - Uncertainty metrics

5. **Platform Channels**
   - **Android**: Kotlin + PyTorch Android Lite
   - **iOS**: Swift + LibTorch (TODO)

## Project Structure

```
retinal_screening/
├── lib/
│   ├── main.dart                    # App entry point with MultiProvider
│   ├── models/
│   │   └── disease_prediction.dart  # Data models
│   ├── providers/
│   │   └── analysis_provider.dart   # State management
│   ├── services/
│   │   └── model_service.dart       # PyTorch Mobile integration
│   ├── screens/
│   │   ├── home_screen.dart         # Main UI
│   │   └── results_screen.dart      # Results display
│   └── widgets/                     # Reusable components (optional)
├── assets/
│   ├── models/
│   │   └── best_model_mobile.pth    # 119 MB PyTorch model
│   └── data/
│       └── disease_names.json       # 45 disease mappings
├── android/
│   └── app/
│       ├── build.gradle.kts         # PyTorch dependencies
│       └── src/main/kotlin/.../
│           └── MainActivity.kt      # Platform channel implementation
└── pubspec.yaml                     # Dependencies & assets
```

## Setup Instructions

### Prerequisites

1. **Flutter SDK** (3.9.2 or higher)
   ```bash
   flutter --version
   ```

2. **Android Studio** (for Android development)
   - Android SDK 21+ (minSdk)
   - Android NDK (for PyTorch Mobile)

3. **Xcode** (for iOS development, macOS only)
   - iOS 12.0+
   - CocoaPods

### Installation Steps

#### 1. Install Dependencies

```bash
cd /home/darkhorse/Downloads/MLOPS_V1/retinal_screening
flutter pub get
```

This installs:
- `pytorch_lite: ^4.2.0` - PyTorch Mobile for Flutter
- `image_picker: ^1.0.7` - Image selection
- `image: ^4.1.7` - Image processing
- `path_provider: ^2.1.2` - File system access
- `provider: ^6.1.1` - State management
- `fl_chart: ^0.66.2` - Charts
- `percent_indicator: ^4.2.3` - Progress indicators
- `shimmer: ^3.0.0` - Loading animations
- `http: ^1.2.1` - API calls (optional)

#### 2. Verify Assets

Ensure these files exist:
```bash
ls -lh assets/models/best_model_mobile.pth     # Should be ~119 MB
ls -lh assets/data/disease_names.json          # Should contain 45 diseases
```

#### 3. Android Setup

The Android configuration is already complete:
- `android/app/build.gradle.kts` includes PyTorch dependencies
- `MainActivity.kt` implements model loading and inference
- No additional steps required

#### 4. iOS Setup (TODO)

iOS platform channel needs implementation:

Create `ios/Runner/ModelPlugin.swift`:
```swift
import Flutter
import UIKit
import LibTorch

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    var module: TorchModule?
    
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let controller = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(
            name: "com.retinal.screening/model",
            binaryMessenger: controller.binaryMessenger
        )
        
        channel.setMethodCallHandler { [weak self] (call, result) in
            switch call.method {
            case "initModel":
                self?.initModel(result: result)
            case "runInference":
                if let args = call.arguments as? [String: Any],
                   let imageData = args["imageData"] as? FlutterStandardTypedData {
                    self?.runInference(imageData: imageData.data, result: result)
                } else {
                    result(FlutterError(code: "INVALID_ARGUMENT", message: nil, details: nil))
                }
            default:
                result(FlutterMethodNotImplemented)
            }
        }
        
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
    
    func initModel(result: @escaping FlutterResult) {
        // TODO: Implement LibTorch model loading
        result(true)
    }
    
    func runInference(imageData: Data, result: @escaping FlutterResult) {
        // TODO: Implement LibTorch inference
        result(["predictions": []])
    }
}
```

Update `ios/Podfile`:
```ruby
platform :ios, '12.0'
pod 'LibTorch-Lite', '~> 1.13.0'
```

## Running the App

### Android

```bash
# Connect Android device or start emulator
flutter devices

# Run in debug mode
flutter run

# Run in release mode (optimized)
flutter run --release
```

### iOS (After iOS setup)

```bash
# Install pods
cd ios && pod install && cd ..

# Run on iOS device/simulator
flutter run
```

## Model Information

### Input Specifications
- **Shape**: [1, 3, 224, 224]
- **Format**: RGB float32 tensor
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Preprocessing**: Automatic in `ModelService.preprocessImage()`

### Output Specifications
- **Shape**: [1, 45]
- **Format**: Sigmoid probabilities (0-1)
- **Diseases**: 45 retinal conditions
- **Post-processing**: Top 5 predictions with confidence levels

### Performance
- **Model Size**: 119 MB (INT8 quantized)
- **Inference Time**: ~200ms on CPU
- **Accuracy**: AUC-ROC 0.64, F1 0.11 (multi-label classification)

## Features

### 1. Image Selection
- Gallery picker
- Camera capture
- Image preview
- Clear/reselect option

### 2. AI Analysis
- Real-time inference
- Progress indicators
- Error handling
- Result caching

### 3. Results Display
- Top 5 disease predictions
- Confidence percentages
- Severity levels (High/Moderate/Low/Very Low)
- Clinical recommendations
- Uncertainty metrics
- Processing time

### 4. UI/UX
- Material 3 design
- Teal primary color (#00897B)
- Responsive layouts
- Loading animations
- Error messages
- Info dialogs

## Disease Classifications

The model detects 45 retinal diseases:

**Major Conditions**:
- DR (Diabetic Retinopathy)
- ARMD (Age-Related Macular Degeneration)
- MH (Macular Hole)
- DN (Diabetic Neuropathy)
- MYA (Myopic Retinopathy)
- BRVO (Branch Retinal Vein Occlusion)
- CRVO (Central Retinal Vein Occlusion)

**Other Conditions**: See `assets/data/disease_names.json` for complete list.

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails
```
Error: Failed to initialize model
```
**Solution**:
- Verify `assets/models/best_model_mobile.pth` exists (119 MB)
- Check `pubspec.yaml` assets configuration
- Run `flutter clean && flutter pub get`

#### 2. Image Processing Fails
```
Error: Failed to decode image
```
**Solution**:
- Ensure image is valid format (JPG, PNG)
- Check file permissions
- Verify image picker permissions in AndroidManifest.xml/Info.plist

#### 3. PyTorch Native Error
```
Error: pytorch_android not found
```
**Solution**:
- Sync Android Gradle files
- Verify `build.gradle.kts` has PyTorch dependencies
- Rebuild: `flutter clean && flutter build apk`

#### 4. Out of Memory
```
Error: OutOfMemoryError
```
**Solution**:
- Model is 119 MB - ensure device has sufficient RAM
- Close other apps
- Use release build (optimized)

### Performance Optimization

1. **Reduce Image Size**:
   ```dart
   final XFile? image = await _picker.pickImage(
     source: source,
     maxWidth: 1024,  // Reduce if needed
     maxHeight: 1024,
     imageQuality: 85, // Lower quality
   );
   ```

2. **Enable ProGuard** (Android):
   Add to `android/app/build.gradle.kts`:
   ```kotlin
   buildTypes {
       release {
           minifyEnabled = true
           proguardFiles(getDefaultProguardFile("proguard-android.txt"), "proguard-rules.pro")
       }
   }
   ```

3. **Lazy Loading**:
   Model initialization happens on first use, not app startup.

## Testing

### Unit Tests (TODO)
```bash
flutter test
```

### Integration Tests (TODO)
```bash
flutter drive --target=test_driver/app.dart
```

### Manual Testing Checklist

- [ ] App launches without errors
- [ ] Model initializes successfully
- [ ] Gallery image selection works
- [ ] Camera capture works
- [ ] Image preview displays correctly
- [ ] Analysis starts with loading indicator
- [ ] Results screen shows predictions
- [ ] Confidence percentages are accurate
- [ ] Back button returns to home
- [ ] Clear image button works
- [ ] App handles errors gracefully

## Deployment

### Android APK

```bash
# Build release APK
flutter build apk --release

# Output: build/app/outputs/flutter-apk/app-release.apk
```

### Android App Bundle (for Play Store)

```bash
# Build app bundle
flutter build appbundle --release

# Output: build/app/outputs/bundle/release/app-release.aab
```

### iOS IPA (macOS only)

```bash
# Build iOS release
flutter build ios --release

# Archive in Xcode for TestFlight/App Store
```

## Next Steps

### Enhancements (Optional)

1. **Explainability Integration**:
   - Use `model_explainer.py` to generate GradCAM heatmaps
   - Add visual overlays showing attention regions
   - Implement server-side explainability endpoint

2. **Batch Processing**:
   - Support multiple image analysis
   - History screen with past results

3. **Cloud Integration**:
   - Fallback to server-side inference
   - Model versioning and updates
   - Usage analytics

4. **Offline Support**:
   - Cache results locally
   - SQLite database for history

5. **Accessibility**:
   - Screen reader support
   - High contrast mode
   - Font size adjustments

6. **Testing**:
   - Unit tests for services/providers
   - Widget tests for UI
   - Integration tests for workflows

## Support

### Resources
- [Flutter Documentation](https://flutter.dev/docs)
- [PyTorch Mobile](https://pytorch.org/mobile)
- [Provider Package](https://pub.dev/packages/provider)

### Model Details
- Location: `/home/darkhorse/Downloads/MLOPS_V1/models/`
- Metadata: `model_metadata.json`
- Explainer: `model_explainer.py`
- Examples: `explainability_examples.py`

## License

[Your License Here]

## Contributors

[Your Name/Team]

---

**Last Updated**: 2024
**Flutter Version**: 3.9.2
**Model Version**: 1.0.0
