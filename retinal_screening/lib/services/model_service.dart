import 'dart:io';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/disease_prediction.dart';

class ModelService {
  // Platform channel for native Android communication
  static const platform = MethodChannel('com.retinal.screening/model');
  
  bool _isInitialized = false;
  bool _useTFLite = false;  // Flag for TFLite availability
  Interpreter? _tfLiteInterpreter;  // TFLite model interpreter
  Map<String, String>? _diseaseNames;
  List<String>? _diseaseCodes;  // Ordered list of disease codes for index mapping
  
  static const String modelAssetTflite = 'assets/models/ai_edge_versions/model_graphclip_rank1_ai_edge.tflite';  // GraphCLIP AI Edge model
  static const String modelAssetPth = 'assets/models/best_model_mobile.pth';  // Fallback: PyTorch
  
  // Initialize the model - TRY TFLite first, fallback to native PyTorch
  Future<void> initialize() async {
    print('🚀 [INIT] Starting model initialization...');
    if (_isInitialized) {
      print('✅ [INIT] Model already initialized');
      return;
    }
    
    try {
      // Load disease names mapping
      print('📚 [INIT] Loading disease names from JSON...');
      final String diseaseJson = await rootBundle.loadString('assets/data/disease_names.json');
      _diseaseNames = Map<String, String>.from(json.decode(diseaseJson));
      print('✅ [INIT] Loaded ${_diseaseNames!.length} disease names');
      
      // Create ordered list of disease codes for index mapping
      _diseaseCodes = _diseaseNames!.keys.toList();
      print('✅ [INIT] Created disease codes list: ${_diseaseCodes!.length} entries');
      
      // Try loading TFLite model first (RECOMMENDED)
      try {
        print('🤖 [INIT] Attempting to load TFLite model...');
        await _loadTFLiteModel();
        _useTFLite = true;
        _isInitialized = true;
        print('✅ [INIT] TFLite model loaded successfully!');
        return;
      } catch (tfliteError) {
        print('⚠️ [INIT] TFLite loading failed: $tfliteError');
        print('🔄 [INIT] Falling back to native PyTorch...');
        // Fallback to native PyTorch via platform channel
      }
      
      // Fallback: Use native PyTorch via platform channel
      try {
        print('🔧 [INIT] Initializing native PyTorch model...');
        final result = await platform.invokeMethod('initModel');
        
        if (result is Map) {
          if (result['success'] == true) {
            _useTFLite = false;
            _isInitialized = true;
            print('✅ [INIT] Native PyTorch model initialized successfully!');
          } else {
            print('❌ [INIT] Native initialization failed: ${result['message'] ?? 'Unknown error'}');
            throw Exception('Model initialization failed: ${result['message'] ?? 'Unknown error'}');
          }
        } else if (result == true) {
          _useTFLite = false;
          _isInitialized = true;
          print('✅ [INIT] Native PyTorch model initialized successfully!');
        } else {
          print('❌ [INIT] Native initialization returned unexpected result: $result');
          throw Exception('Model initialization failed');
        }
      } catch (nativeError) {
        print('❌ [INIT] Native initialization failed: $nativeError');
        throw Exception('Both TFLite and native model initialization failed');
      }
    } catch (e, stackTrace) {
      print('❌ [INIT] Initialization error: $e');
      print('📍 [INIT] Stack trace:\n$stackTrace');
      throw Exception('Error initializing model: $e');
    }
  }
  
  /// Load TensorFlow Lite model from assets
  /// This is the RECOMMENDED approach for Flutter apps
  Future<void> _loadTFLiteModel() async {
    try {
      print('📦 [TFLITE] Loading model from asset: $modelAssetTflite');
      _tfLiteInterpreter = await Interpreter.fromAsset(modelAssetTflite);
      
      // Log model details
      final inputTensors = _tfLiteInterpreter!.getInputTensors();
      final outputTensors = _tfLiteInterpreter!.getOutputTensors();
      
      print('✅ [TFLITE] Model loaded successfully');
      print('📊 [TFLITE] Input tensors: ${inputTensors.length}');
      if (inputTensors.isNotEmpty) {
        print('   - Shape: ${inputTensors[0].shape}');
        print('   - Type: ${inputTensors[0].type}');
      }
      print('📊 [TFLITE] Output tensors: ${outputTensors.length}');
      if (outputTensors.isNotEmpty) {
        print('   - Shape: ${outputTensors[0].shape}');
        print('   - Type: ${outputTensors[0].type}');
      }
    } catch (e, stackTrace) {
      print('❌ [TFLITE] Failed to load model: $e');
      print('📍 [TFLITE] Stack trace:\n$stackTrace');
      throw Exception('Failed to load TFLite model: $e');
    }
  }
  
  // Preprocess image for model input
  Future<Uint8List> preprocessImage(File imageFile) async {
    print('🖼️ [PREPROCESS] Starting image preprocessing in isolate...');
    // Run preprocessing in a separate isolate to avoid blocking UI
    final result = await compute(_preprocessImageIsolate, imageFile.path);
    print('✅ [PREPROCESS] Preprocessing completed');
    return result;
  }
  
  // Static method for isolate-based preprocessing
  static Future<Uint8List> _preprocessImageIsolate(String imagePath) async {
    print('🔄 [ISOLATE] Loading image from: $imagePath');
    // Load image
    final bytes = await File(imagePath).readAsBytes();
    print('✅ [ISOLATE] Image loaded: ${bytes.length} bytes');
    
    img.Image? image = img.decodeImage(bytes);
    
    if (image == null) {
      print('❌ [ISOLATE] Failed to decode image');
      throw Exception('Failed to decode image');
    }
    
    print('✅ [ISOLATE] Image decoded: ${image.width}x${image.height}');
    
    // Resize to 224x224
    img.Image resized = img.copyResize(image, width: 224, height: 224);
    print('✅ [ISOLATE] Image resized to 224x224');
    
    // Convert to RGB and normalize - Output in NCHW format for PyTorch
    List<double> normalizedR = [];
    List<double> normalizedG = [];
    List<double> normalizedB = [];
    
    // ImageNet normalization constants
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        final pixel = resized.getPixel(x, y);
        
        // Normalize each channel
        final r = (pixel.r / 255.0 - mean[0]) / std[0];
        final g = (pixel.g / 255.0 - mean[1]) / std[1];
        final b = (pixel.b / 255.0 - mean[2]) / std[2];
        
        normalizedR.add(r);
        normalizedG.add(g);
        normalizedB.add(b);
      }
    }
    
    print('✅ [ISOLATE] Image normalized (ImageNet stats)');
    
    // Combine channels in NCHW order: [R, G, B]
    List<double> normalized = normalizedR + normalizedG + normalizedB;
    
    // Convert to Float32List
    final Float32List float32List = Float32List.fromList(normalized);
    
    print('✅ [ISOLATE] Converted to Float32List: ${float32List.length} values');
    
    // Convert to Uint8List for platform channel
    return float32List.buffer.asUint8List();
  }
  
  // Run inference on image
  Future<AnalysisResult> analyzeImage(File imageFile) async {
    print('🔬 [MODEL] Starting analyzeImage...');
    if (!_isInitialized) {
      print('⚠️ [MODEL] Model not initialized, initializing now...');
      await initialize();
    }
    
    print('✅ [MODEL] Model initialized. Using TFLite: $_useTFLite');
    
    try {
      final startTime = DateTime.now();
      print('📸 [MODEL] Preprocessing image: ${imageFile.path}');
      
      // Preprocess image
      final preprocessed = await preprocessImage(imageFile);
      print('✅ [MODEL] Image preprocessed. Size: ${preprocessed.length} bytes');
      
      List<dynamic> predictions;
      
      if (_useTFLite && _tfLiteInterpreter != null) {
        print('🤖 [MODEL] Running TFLite inference...');
        // Use TFLite for inference (PREFERRED)
        predictions = await _runTFLiteInference(preprocessed);
        print('✅ [MODEL] TFLite inference completed. Predictions: ${predictions.length}');
      } else {
        print('🔄 [MODEL] Running native PyTorch inference...');
        // Fallback to native PyTorch
        predictions = await _runNativeInference(preprocessed);
        print('✅ [MODEL] Native inference completed. Predictions: ${predictions.length}');
      }
      
      final inferenceTime = DateTime.now().difference(startTime).inMilliseconds.toDouble();
      print('⏱️ [MODEL] Total inference time: ${inferenceTime}ms');
      
      // Parse results
      print('📊 [MODEL] Parsing predictions...');
      final List<DiseasePrediction> topPredictions = [];
      
      // Get top 5 predictions
      List<MapEntry<int, double>> indexedPreds = [];
      for (int i = 0; i < predictions.length; i++) {
        indexedPreds.add(MapEntry(i, predictions[i].toDouble()));
      }
      
      indexedPreds.sort((a, b) => b.value.compareTo(a.value));
      print('🎯 [MODEL] Top 5 predictions:');
      
      for (int i = 0; i < 5 && i < indexedPreds.length; i++) {
        final entry = indexedPreds[i];
        final diseaseCode = _getDiseaseCode(entry.key);
        final diseaseName = _diseaseNames?[diseaseCode] ?? diseaseCode;
        
        print('   ${i + 1}. $diseaseName ($diseaseCode): ${(entry.value * 100).toStringAsFixed(2)}%');
        
        topPredictions.add(DiseasePrediction(
          diseaseCode: diseaseCode,
          diseaseName: diseaseName,
          confidence: entry.value,
          severity: _getSeverity(entry.value),
          recommendation: _getRecommendation(entry.value, diseaseName),
        ));
      }
      
      // Calculate uncertainty (entropy)
      double uncertainty = _calculateUncertainty(predictions);
      print('📈 [MODEL] Uncertainty score: ${uncertainty.toStringAsFixed(3)}');
      
      print('✅ [MODEL] Analysis complete!');
      return AnalysisResult(
        topPredictions: topPredictions,
        inferenceTimeMs: inferenceTime,
        uncertainty: uncertainty,
        modelVersion: '1.0.0',
        timestamp: DateTime.now(),
      );
    } catch (e, stackTrace) {
      print('❌ [MODEL] Analysis failed: $e');
      print('📍 [MODEL] Stack trace:\n$stackTrace');
      throw Exception('Analysis failed: $e');
    }
  }
  
  /// Run inference using TensorFlow Lite
  Future<List<dynamic>> _runTFLiteInference(Uint8List imageData) async {
    try {
      print('🧠 [TFLITE] Starting TFLite inference...');
      if (_tfLiteInterpreter == null) {
        print('❌ [TFLITE] Interpreter is null');
        throw Exception('TFLite interpreter not initialized');
      }
      
      print('📊 [TFLITE] Input data size: ${imageData.length} bytes');
      
      // Prepare input: Convert Uint8List to appropriate format
      // Input shape: [1, 3, 224, 224] for PyTorch TFLite model (NCHW format)
      final Float32List inputBytes = Float32List.fromList(
        imageData.buffer.asFloat32List()
      );
      
      print('📊 [TFLITE] Input tensor size: ${inputBytes.length} floats');
      
      // Create input and output objects
      var input = inputBytes.reshape([1, 3, 224, 224]);
      // Output shape must match model output: [1, 45] (batch_size, num_classes)
      var output = List.generate(1, (_) => List<double>.filled(45, 0.0));
      
      print('🚀 [TFLITE] Running interpreter...');
      print('📊 [TFLITE] Output shape: [${output.length}, ${output[0].length}]');
      
      // Run inference
      _tfLiteInterpreter!.run(input, output);
      
      // Extract predictions from batch dimension
      final predictions = output[0];
      
      print('✅ [TFLITE] Inference completed. Output size: ${predictions.length}');
      
      // Log first few predictions for debugging
      print('📈 [TFLITE] First 5 outputs: ${predictions.take(5).toList()}');
      
      return predictions;
    } catch (e, stackTrace) {
      print('❌ [TFLITE] Inference failed: $e');
      print('📍 [TFLITE] Stack trace:\n$stackTrace');
      throw Exception('TFLite inference failed: $e');
    }
  }
  
  /// Fallback: Run inference using native PyTorch via platform channel
  Future<List<dynamic>> _runNativeInference(Uint8List imageData) async {
    try {
      print('🔧 [NATIVE] Starting native PyTorch inference...');
      print('📊 [NATIVE] Input data size: ${imageData.length} bytes');
      
      final Map<dynamic, dynamic> result = await platform.invokeMethod(
        'runInference',
        {'imageData': imageData},
      );
      
      print('✅ [NATIVE] Native inference completed');
      
      final List<dynamic> predictions = result['predictions'];
      print('📊 [NATIVE] Predictions count: ${predictions.length}');
      print('📈 [NATIVE] First 5 outputs: ${predictions.take(5).toList()}');
      
      return predictions;
    } catch (e, stackTrace) {
      print('❌ [NATIVE] Inference failed: $e');
      print('📍 [NATIVE] Stack trace:\n$stackTrace');
      throw Exception('Native inference failed: $e');
    }
  }
  
  // Helper: Get disease code from index
  String _getDiseaseCode(int index) {
    if (_diseaseCodes != null && index < _diseaseCodes!.length) {
      return _diseaseCodes![index];
    }
    return 'UNKNOWN_$index';
  }
  
  // Helper: Get severity level
  String _getSeverity(double confidence) {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.5) return 'Moderate';
    if (confidence >= 0.3) return 'Low';
    return 'Very Low';
  }
  
  // Helper: Get clinical recommendation
  String _getRecommendation(double confidence, String diseaseName) {
    if (confidence >= 0.8) {
      return 'Immediate consultation with an ophthalmologist recommended. $diseaseName detected with high confidence.';
    } else if (confidence >= 0.5) {
      return 'Schedule an appointment with an eye care professional for further evaluation of possible $diseaseName.';
    } else if (confidence >= 0.3) {
      return 'Consider routine eye examination. Low indicators of $diseaseName detected.';
    } else {
      return 'Continue regular eye health monitoring. Very low confidence detection.';
    }
  }
  
  // Helper: Calculate uncertainty (entropy)
  double _calculateUncertainty(List<dynamic> predictions) {
    double entropy = 0.0;
    for (var pred in predictions) {
      final p = pred.toDouble();
      if (p > 0) {
        entropy -= p * log(p.clamp(1e-10, 1.0)) / 2.302585; // log base 10
      }
    }
    return entropy;
  }

  // Test method to verify model loading and basic functionality
  Future<Map<String, dynamic>> testModelLoading() async {
    if (!_isInitialized) {
      await initialize();
    }

    final Map<String, dynamic> testResults = {
      'model_loaded': _isInitialized,
      'using_tflite': _useTFLite,
      'disease_names_loaded': _diseaseNames != null,
      'disease_count': _diseaseNames?.length ?? 0,
      'model_path': _useTFLite ? modelAssetTflite : modelAssetPth,
    };

    // Test TFLite model details if available
    if (_useTFLite && _tfLiteInterpreter != null) {
      try {
        final inputDetails = _tfLiteInterpreter!.getInputTensors();
        final outputDetails = _tfLiteInterpreter!.getOutputTensors();

        testResults.addAll({
          'input_shape': inputDetails.isNotEmpty ? inputDetails[0].shape : null,
          'output_shape': outputDetails.isNotEmpty ? outputDetails[0].shape : null,
          'input_type': inputDetails.isNotEmpty ? inputDetails[0].type.toString() : null,
          'output_type': outputDetails.isNotEmpty ? outputDetails[0].type.toString() : null,
        });
      } catch (e) {
        testResults['tflite_details_error'] = e.toString();
      }
    }

    // Test disease name mapping
    if (_diseaseCodes != null && _diseaseCodes!.isNotEmpty) {
      testResults.addAll({
        'sample_disease_codes': _diseaseCodes!.take(5).toList(),
        'sample_disease_names': _diseaseCodes!.take(5).map((code) => _diseaseNames![code]).toList(),
      });
    }

    return testResults;
  }
}
