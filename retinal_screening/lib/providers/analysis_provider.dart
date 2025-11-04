import 'dart:io';
import 'package:flutter/material.dart';
import '../models/disease_prediction.dart';

class AnalysisProvider extends ChangeNotifier {
  File? _selectedImage;
  bool _isAnalyzing = false;
  AnalysisResult? _result;
  String? _error;

  File? get selectedImage => _selectedImage;
  bool get isAnalyzing => _isAnalyzing;
  AnalysisResult? get result => _result;
  String? get error => _error;
  bool get hasImage => _selectedImage != null;
  bool get hasResult => _result != null;

  void setImage(File image) {
    _selectedImage = image;
    _result = null; // Clear previous results
    _error = null;
    notifyListeners();
  }

  void clearImage() {
    _selectedImage = null;
    _result = null;
    _error = null;
    notifyListeners();
  }

  void startAnalysis() {
    _isAnalyzing = true;
    _error = null;
    notifyListeners();
  }

  void setResult(AnalysisResult result) {
    _result = result;
    _isAnalyzing = false;
    _error = null;
    notifyListeners();
  }

  void setError(String error) {
    _error = error;
    _isAnalyzing = false;
    notifyListeners();
  }

  void reset() {
    _selectedImage = null;
    _isAnalyzing = false;
    _result = null;
    _error = null;
    notifyListeners();
  }
}
