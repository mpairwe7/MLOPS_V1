#!/usr/bin/env python3
"""
Compare TFLite models: Original vs AI Edge Torch versions
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

def test_tflite_model(model_path, name):
    """Test a TFLite model and return results"""
    try:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Create test input
        test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        return {
            'name': name,
            'input_shape': input_details[0]['shape'],
            'output_shape': output.shape,
            'dtype': output.dtype,
            'sample_predictions': output[0][:5].tolist(),
            'file_size': model_path.stat().st_size / (1024**2),
            'status': 'SUCCESS'
        }

    except Exception as e:
        return {
            'name': name,
            'status': 'FAILED',
            'error': str(e)[:100]
        }

def main():
    print("=" * 80)
    print("TFLite Model Comparison: Original vs AI Edge Torch")
    print("=" * 80)

    models_dir = Path("assets/models")

    # Find all TFLite models
    original_models = list(models_dir.glob("*.tflite"))
    ai_edge_models = list(models_dir.glob("ai_edge_versions/*.tflite"))

    print(f"\nFound {len(original_models)} original models")
    print(f"Found {len(ai_edge_models)} AI Edge models")

    all_results = []

    # Test original models
    print("\n" + "="*50)
    print("TESTING ORIGINAL MODELS")
    print("="*50)

    for model_path in original_models:
        name = f"Original: {model_path.name}"
        result = test_tflite_model(model_path, name)
        all_results.append(result)

        if result['status'] == 'SUCCESS':
            print(f"\n✓ {result['name']}")
            print(f"  Size: {result['file_size']:.2f} MB")
            print(f"  Input: {result['input_shape']}")
            print(f"  Output: {result['output_shape']}")
            print(f"  Sample: {result['sample_predictions']}")
        else:
            print(f"\n✗ {result['name']}: {result['error']}")

    # Test AI Edge models
    print("\n" + "="*50)
    print("TESTING AI EDGE TORCH MODELS")
    print("="*50)

    for model_path in ai_edge_models:
        name = f"AI Edge: {model_path.name}"
        result = test_tflite_model(model_path, name)
        all_results.append(result)

        if result['status'] == 'SUCCESS':
            print(f"\n✓ {result['name']}")
            print(f"  Size: {result['file_size']:.2f} MB")
            print(f"  Input: {result['input_shape']}")
            print(f"  Output: {result['output_shape']}")
            print(f"  Sample: {result['sample_predictions']}")
        else:
            print(f"\n✗ {result['name']}: {result['error']}")

    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    successful_models = [r for r in all_results if r['status'] == 'SUCCESS']

    if len(successful_models) >= 2:
        # Group by base name
        model_groups = {}
        for result in successful_models:
            base_name = result['name'].split(': ')[1].replace('_ai_edge.tflite', '').replace('.tflite', '')
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(result)

        for base_name, models in model_groups.items():
            if len(models) == 2:
                orig, ai_edge = models
                size_diff = ai_edge['file_size'] - orig['file_size']

                print(f"\n📊 {base_name}:")
                print(".2f"                print(".2f"                print(".2f"
                # Compare predictions if same shape
                if orig['output_shape'] == ai_edge['output_shape']:
                    pred_diff = np.abs(np.array(orig['sample_predictions']) - np.array(ai_edge['sample_predictions']))
                    print(".4f"    else:
        print(f"\nTotal models tested: {len(all_results)}")
        print(f"Successful: {len(successful_models)}")
        print(f"Failed: {len(all_results) - len(successful_models)}")

if __name__ == "__main__":
    # Change to project root
    os.chdir(Path(__file__).parent)
    main()