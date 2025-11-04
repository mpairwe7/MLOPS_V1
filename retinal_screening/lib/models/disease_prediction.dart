class DiseasePrediction {
  final String diseaseCode;
  final String diseaseName;
  final double confidence;
  final String severity;
  final String recommendation;

  DiseasePrediction({
    required this.diseaseCode,
    required this.diseaseName,
    required this.confidence,
    required this.severity,
    required this.recommendation,
  });

  String get confidencePercentage => '${(confidence * 100).toStringAsFixed(1)}%';

  String get severityColor {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.5) return 'medium';
    return 'low';
  }

  Map<String, dynamic> toJson() => {
        'diseaseCode': diseaseCode,
        'diseaseName': diseaseName,
        'confidence': confidence,
        'severity': severity,
        'recommendation': recommendation,
      };

  factory DiseasePrediction.fromJson(Map<String, dynamic> json) {
    return DiseasePrediction(
      diseaseCode: json['diseaseCode'] as String,
      diseaseName: json['diseaseName'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      severity: json['severity'] as String,
      recommendation: json['recommendation'] as String,
    );
  }
}

class AnalysisResult {
  final List<DiseasePrediction> topPredictions;
  final double inferenceTimeMs;
  final double uncertainty;
  final String modelVersion;
  final DateTime timestamp;

  AnalysisResult({
    required this.topPredictions,
    required this.inferenceTimeMs,
    required this.uncertainty,
    required this.modelVersion,
    required this.timestamp,
  });

  String get formattedTimestamp =>
      '${timestamp.day}/${timestamp.month}/${timestamp.year} ${timestamp.hour}:${timestamp.minute.toString().padLeft(2, '0')}';

  Map<String, dynamic> toJson() => {
        'topPredictions': topPredictions.map((p) => p.toJson()).toList(),
        'inferenceTimeMs': inferenceTimeMs,
        'uncertainty': uncertainty,
        'modelVersion': modelVersion,
        'timestamp': timestamp.toIso8601String(),
      };
}
