// Basic widget tests for Retinal AI Screening App
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:retinal_screening/main.dart';

void main() {
  testWidgets('App launches with correct title', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const RetinalScreeningApp());

    // Verify that the app bar has the correct title
    expect(find.text('Retinal AI Screening'), findsOneWidget);

    // Verify that the header text is present
    expect(find.text('AI-Powered Retinal Disease Detection'), findsOneWidget);
  });

  testWidgets('Home screen displays correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const RetinalScreeningApp());
    await tester.pumpAndSettle();

    // Verify key UI elements are present
    expect(find.byIcon(Icons.visibility), findsOneWidget);
    expect(find.text('Select Retinal Image'), findsOneWidget);
    expect(find.text('How it works'), findsOneWidget);
  });

  testWidgets('Info dialog can be opened', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const RetinalScreeningApp());
    await tester.pumpAndSettle();

    // Find and tap the info icon
    final infoButton = find.byIcon(Icons.info_outline);
    expect(infoButton, findsOneWidget);
    
    await tester.tap(infoButton);
    await tester.pumpAndSettle();

    // Verify the dialog appears with correct content
    expect(find.text('About This App'), findsOneWidget);
    expect(find.textContaining('47 different retinal diseases'), findsOneWidget);
  });
}
