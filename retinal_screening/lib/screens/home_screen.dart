import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import '../providers/analysis_provider.dart';
import '../services/model_service.dart';
import 'results_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  Map<String, dynamic>? _modelTestResults;
  
  @override
  void initState() {
    super.initState();
    // Initialize model on startup
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeModel();
    });
  }
  
  Future<void> _initializeModel() async {
    final modelService = Provider.of<ModelService>(context, listen: false);
    
    // Wait a bit to ensure Scaffold is fully built
    await Future.delayed(const Duration(milliseconds: 100));
    
    try {
      // Show loading indicator
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Row(
              children: [
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  ),
                ),
                SizedBox(width: 12),
                Text('Loading AI Model...'),
              ],
            ),
            backgroundColor: Colors.blue,
            duration: Duration(seconds: 5),
          ),
        );
      }
      
      await modelService.initialize();
      
      if (mounted) {
        // Clear loading message and show success
        ScaffoldMessenger.of(context).clearSnackBars();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Row(
              children: [
                Icon(Icons.check_circle, color: Colors.white),
                SizedBox(width: 12),
                Text('AI Model loaded successfully'),
              ],
            ),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        // Clear loading message and show error
        ScaffoldMessenger.of(context).clearSnackBars();
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.error, color: Colors.white),
                const SizedBox(width: 12),
                Expanded(child: Text('Failed to load model: $e')),
              ],
            ),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }

  Future<void> _testModel() async {
    final modelService = Provider.of<ModelService>(context, listen: false);
    try {
      final results = await modelService.testModelLoading();
      setState(() {
        _modelTestResults = results;
      });
      
      // Results are now displayed in the main UI
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Model test failed: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }
  
  Future<void> _pickImage(ImageSource source) async {
    final provider = Provider.of<AnalysisProvider>(context, listen: false);
    
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 90,
      );
      
      if (image != null) {
        provider.setImage(File(image.path));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick image: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
  
  Future<void> _analyzeImage() async {
    print('🔍 [DEBUG] Starting image analysis...');
    final provider = Provider.of<AnalysisProvider>(context, listen: false);
    final modelService = Provider.of<ModelService>(context, listen: false);
    
    if (provider.selectedImage == null) {
      print('❌ [DEBUG] No image selected');
      return;
    }
    
    print('✅ [DEBUG] Image path: ${provider.selectedImage!.path}');
    print('📊 [DEBUG] Starting analysis provider...');
    provider.startAnalysis();
    
    try {
      print('🤖 [DEBUG] Calling model service analyzeImage...');
      final result = await modelService.analyzeImage(provider.selectedImage!);
      print('✅ [DEBUG] Analysis completed successfully!');
      print('📈 [DEBUG] Inference time: ${result.inferenceTimeMs}ms');
      print('🎯 [DEBUG] Top prediction: ${result.topPredictions.first.diseaseName} (${result.topPredictions.first.confidencePercentage})');
      
      provider.setResult(result);
      
      if (mounted) {
        print('🚀 [DEBUG] Navigating to results screen...');
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => const ResultsScreen()),
        );
      }
    } catch (e, stackTrace) {
      print('❌ [DEBUG] Analysis failed with error: $e');
      print('📍 [DEBUG] Stack trace:\n$stackTrace');
      
      provider.setError('Analysis failed: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Analysis failed: $e'),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    }
  }
  
  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library, color: Color(0xFF00897B)),
              title: const Text('Choose from Gallery'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.gallery);
              },
            ),
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Color(0xFF00897B)),
              title: const Text('Take a Photo'),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
          ],
        ),
      ),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Retinal AI Screening'),
        actions: [
          IconButton(
            icon: const Icon(Icons.science),
            tooltip: 'Test Model',
            onPressed: _testModel,
          ),
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: () => _showInfoDialog(),
          ),
        ],
      ),
      body: Consumer<AnalysisProvider>(
        builder: (context, provider, child) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header
                Card(
                  color: Theme.of(context).colorScheme.primaryContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        Icon(
                          Icons.visibility,
                          size: 48,
                          color: Theme.of(context).colorScheme.primary,
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'AI-Powered Retinal Disease Detection',
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Upload a retinal image for instant AI analysis',
                          style: Theme.of(context).textTheme.bodyMedium,
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                
                // Image display
                if (provider.hasImage) ...[
                  Card(
                    clipBehavior: Clip.antiAlias,
                    child: Column(
                      children: [
                        Image.file(
                          provider.selectedImage!,
                          height: 300,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                        Padding(
                          padding: const EdgeInsets.all(12),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                'Selected Image',
                                style: Theme.of(context).textTheme.titleMedium,
                              ),
                              TextButton.icon(
                                onPressed: provider.clearImage,
                                icon: const Icon(Icons.close),
                                label: const Text('Clear'),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                ],
                
                // Select image button
                if (!provider.hasImage)
                  OutlinedButton.icon(
                    onPressed: _showImageSourceDialog,
                    icon: const Icon(Icons.add_photo_alternate, size: 32),
                    label: const Padding(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      child: Text(
                        'Select Retinal Image',
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: BorderSide(
                        color: Theme.of(context).colorScheme.primary,
                        width: 2,
                      ),
                    ),
                  ),
                
                // Analyze button
                if (provider.hasImage && !provider.isAnalyzing)
                  ElevatedButton.icon(
                    onPressed: _analyzeImage,
                    icon: const Icon(Icons.analytics),
                    label: const Padding(
                      padding: EdgeInsets.symmetric(vertical: 16),
                      child: Text(
                        'Analyze with AI',
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Theme.of(context).colorScheme.primary,
                      foregroundColor: Colors.white,
                    ),
                  ),
                
                // Loading indicator
                if (provider.isAnalyzing) ...[
                  const SizedBox(height: 20),
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        children: [
                          const CircularProgressIndicator(),
                          const SizedBox(height: 16),
                          Text(
                            'Analyzing retinal image...',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'This may take a few seconds',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
                
                // Model test results display
                if (_modelTestResults != null) ...[
                  const SizedBox(height: 20),
                  Card(
                    color: Colors.blue.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.science,
                                color: Theme.of(context).colorScheme.primary,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Model Test Results',
                                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                      fontWeight: FontWeight.bold,
                                    ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 12),
                          _buildTestResultRow('Model Loaded', _modelTestResults!['model_loaded'] ?? false),
                          _buildTestResultRow('Using TFLite', _modelTestResults!['using_tflite'] ?? false),
                          _buildTestResultRow('Disease Names Loaded', _modelTestResults!['disease_names_loaded'] ?? false),
                          const SizedBox(height: 8),
                          Text(
                            'Disease Count: ${_modelTestResults!['disease_count'] ?? 0}',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                          Text(
                            'Model Path: ${_modelTestResults!['model_path'] ?? 'Unknown'}',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                          
                          if (_modelTestResults!.containsKey('input_shape')) ...[
                            const SizedBox(height: 8),
                            Text(
                              'Input Shape: ${_modelTestResults!['input_shape']}',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                            Text(
                              'Output Shape: ${_modelTestResults!['output_shape']}',
                              style: Theme.of(context).textTheme.bodySmall,
                            ),
                          ],
                          
                          if (_modelTestResults!.containsKey('sample_disease_codes')) ...[
                            const SizedBox(height: 12),
                            Text(
                              'Sample Disease Mappings:',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                            const SizedBox(height: 4),
                            ...(_modelTestResults!['sample_disease_codes'] as List<dynamic>).asMap().entries.take(3).map((entry) {
                              final code = entry.value;
                              final name = (_modelTestResults!['sample_disease_names'] as List<dynamic>)[entry.key];
                              return Text(
                                '• $code → $name',
                                style: Theme.of(context).textTheme.bodySmall,
                              );
                            }),
                          ],
                          
                          if (_modelTestResults!.containsKey('tflite_details_error')) ...[
                            const SizedBox(height: 8),
                            Text(
                              'TFLite Details Error: ${_modelTestResults!['tflite_details_error']}',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: Colors.red,
                                  ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                ],
                
                // Info cards
                if (!provider.hasImage) ...[
                  const SizedBox(height: 32),
                ],
              ],
            ),
          );
        },
      ),
    );
  }
  

  
  void _showInfoDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About This App'),
        content: const SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'This application uses advanced AI to detect retinal diseases from fundus images.',
                style: TextStyle(fontSize: 16),
              ),
              SizedBox(height: 16),
              Text(
                'Features:',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              SizedBox(height: 8),
              Text('• Detection of 47 different retinal diseases'),
              Text('• High-accuracy deep learning model'),
              Text('• Confidence scores for predictions'),
              Text('• Clinical recommendations'),
              SizedBox(height: 16),
              Text(
                '⚠️ Important: This is a screening tool and not a replacement for professional medical diagnosis.',
                style: TextStyle(fontStyle: FontStyle.italic, color: Colors.orange),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Got it'),
          ),
        ],
      ),
    );
  }

  Widget _buildTestResultRow(String label, bool success) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          Icon(
            success ? Icons.check_circle : Icons.error,
            color: success ? Colors.green : Colors.red,
            size: 16,
          ),
          const SizedBox(width: 8),
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}
