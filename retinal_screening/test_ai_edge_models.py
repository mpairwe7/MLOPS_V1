#!/usr/bin/env python3
"""
Test AI Edge Torch Converted TFLite Models
Verifies that the converted models load and run inference correctly
"""

import os
import numpy as np
import tensorflow as tf

def test_tflite_model(model_path, model_name):
    """Test a single TFLite model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        # Load the model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"✓ Model loaded successfully")
        print(f"✓ Input shape: {input_details[0]['shape']}")
        print(f"✓ Output shape: {output_details[0]['shape']}")

        # Create test input (random image)
        test_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"✓ Inference successful")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Check output properties
        assert output.shape == (1, 45), f"Expected output shape (1, 45), got {output.shape}"
        assert np.isfinite(output).all(), "Output contains NaN or Inf values"

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def main():
    """Test all AI Edge Torch converted models"""
    print("AI Edge Torch TFLite Model Testing")
    print("=" * 60)

    # Models to test
    models_dir = "ai_edge_versions"
    models = [
        ("best_model_mobile_ai_edge.tflite", "Best Model Mobile"),
        ("model_graphclip_rank1_ai_edge.tflite", "GraphCLIP Rank 1")
    ]

    results = []

    for model_file, model_name in models:
        model_path = os.path.join(models_dir, model_file)

        if os.path.exists(model_path):
            success = test_tflite_model(model_path, model_name)
            results.append((model_name, success))
        else:
            print(f"\n✗ Model not found: {model_path}")
            results.append((model_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = 0
    for model_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name}: {status}")
        if success:
            passed += 1

    print(f"\nPassed: {passed}/{len(results)} models")

    if passed == len(results):
        print("🎉 All AI Edge Torch models are working correctly!")
        return 0
    else:
        print("⚠️  Some models failed testing")
        return 1

if __name__ == "__main__":
    exit(main())