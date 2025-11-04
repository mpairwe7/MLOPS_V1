import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:percent_indicator/percent_indicator.dart';
import '../providers/analysis_provider.dart';
import '../models/disease_prediction.dart';

class ResultsScreen extends StatelessWidget {
  const ResultsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<AnalysisProvider>(context);
    final result = provider.result;

    if (result == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Results')),
        body: const Center(child: Text('No results available')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () => _shareResults(context, result),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Summary Card
            Card(
              color: Theme.of(context).colorScheme.primaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    Icon(
                      Icons.check_circle,
                      size: 64,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Analysis Complete',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      result.formattedTimestamp,
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildStatItem(
                          context,
                          'Processing Time',
                          '${result.inferenceTimeMs.toStringAsFixed(0)}ms',
                        ),
                        _buildStatItem(
                          context,
                          'Model Version',
                          result.modelVersion,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Top Prediction Highlight
            Text(
              'Top Detection',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            _buildTopPredictionCard(context, result.topPredictions.first),
            const SizedBox(height: 24),

            // All Predictions
            Text(
              'All Detections',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            ...result.topPredictions.map((pred) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: _buildPredictionCard(context, pred),
                )),

            // Uncertainty Indicator
            const SizedBox(height: 24),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.info_outline, size: 20),
                        const SizedBox(width: 8),
                        Text(
                          'Model Confidence Metrics',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    LinearPercentIndicator(
                      lineHeight: 20,
                      percent: (1 - result.uncertainty / 10).clamp(0.0, 1.0),
                      center: Text(
                        'Certainty: ${((1 - result.uncertainty / 10) * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(fontSize: 12, color: Colors.white),
                      ),
                      backgroundColor: Colors.grey[300],
                      progressColor: _getUncertaintyColor(result.uncertainty),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _getUncertaintyDescription(result.uncertainty),
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            ),

            // Disclaimer
            const SizedBox(height: 24),
            Card(
              color: Colors.orange.shade50,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.warning_amber, color: Colors.orange),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'This AI screening tool is for informational purposes only. '
                        'Always consult with a qualified healthcare professional for proper diagnosis and treatment.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () {
                provider.reset();
                Navigator.pop(context);
              },
              icon: const Icon(Icons.refresh),
              label: const Padding(
                padding: EdgeInsets.symmetric(vertical: 16),
                child: Text('Analyze Another Image'),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).colorScheme.primary,
                foregroundColor: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(BuildContext context, String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }

  Widget _buildTopPredictionCard(BuildContext context, DiseasePrediction pred) {
    return Card(
      elevation: 4,
      color: _getConfidenceColor(pred.confidence).withValues(alpha: 0.1),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.medical_information,
                  color: _getConfidenceColor(pred.confidence),
                  size: 32,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        pred.diseaseName,
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      Text(
                        pred.diseaseCode,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            CircularPercentIndicator(
              radius: 60,
              lineWidth: 12,
              percent: pred.confidence,
              center: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    pred.confidencePercentage,
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    pred.severity,
                    style: const TextStyle(fontSize: 12),
                  ),
                ],
              ),
              progressColor: _getConfidenceColor(pred.confidence),
              backgroundColor: Colors.grey[300]!,
            ),
            const SizedBox(height: 16),
            const Divider(),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.recommend, size: 20),
                const SizedBox(width: 8),
                Text(
                  'Recommendation',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              pred.recommendation,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionCard(BuildContext context, DiseasePrediction pred) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        pred.diseaseName,
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      Text(
                        pred.diseaseCode,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: _getConfidenceColor(pred.confidence).withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    pred.confidencePercentage,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: _getConfidenceColor(pred.confidence),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            LinearPercentIndicator(
              lineHeight: 8,
              percent: pred.confidence,
              backgroundColor: Colors.grey[300],
              progressColor: _getConfidenceColor(pred.confidence),
              barRadius: const Radius.circular(4),
            ),
            const SizedBox(height: 8),
            Text(
              'Severity: ${pred.severity}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.red;
    if (confidence >= 0.5) return Colors.orange;
    if (confidence >= 0.3) return Colors.amber;
    return Colors.green;
  }

  Color _getUncertaintyColor(double uncertainty) {
    if (uncertainty < 2) return Colors.green;
    if (uncertainty < 4) return Colors.orange;
    return Colors.red;
  }

  String _getUncertaintyDescription(double uncertainty) {
    if (uncertainty < 2) {
      return 'High confidence: The model is very certain about these predictions.';
    } else if (uncertainty < 4) {
      return 'Moderate confidence: Multiple conditions may be present or image quality may vary.';
    } else {
      return 'Lower confidence: Consider retaking the image or consulting a specialist.';
    }
  }

  void _shareResults(BuildContext context, AnalysisResult result) {
    // TODO: Implement share functionality
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Share functionality coming soon')),
    );
  }
}
