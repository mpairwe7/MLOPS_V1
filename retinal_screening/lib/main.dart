import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/home_screen.dart';
import 'services/model_service.dart';
import 'providers/analysis_provider.dart';

void main() {
  runApp(const RetinalScreeningApp());
}

class RetinalScreeningApp extends StatelessWidget {
  const RetinalScreeningApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AnalysisProvider()),
        Provider(create: (_) => ModelService()),
      ],
      child: MaterialApp(
        title: 'Retinal AI Screening',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.teal,
          primaryColor: const Color(0xFF00897B),
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF00897B),
            brightness: Brightness.light,
          ),
          useMaterial3: true,
          cardTheme: CardThemeData(
            elevation: 2,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          appBarTheme: const AppBarTheme(
            centerTitle: true,
            elevation: 0,
          ),
        ),
        home: const HomeScreen(),
      ),
    );
  }
}
