#!/usr/bin/env python3
"""
Test script to verify TFLite model outputs with different inputs
This will help diagnose if the model is always producing the same predictions
"""

import numpy as np
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
from pathlib import Path

# Load the TFLite model
model_path = "assets/models/ai_edge_versions/model_graphclip_rank1_ai_edge.tflite"

print(f"🔍 Loading model from: {model_path}")
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n📊 Model Details:")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output type: {output_details[0]['dtype']}")

# Load disease names
import json
with open('assets/data/disease_names.json', 'r') as f:
    disease_names = json.load(f)
    disease_codes = list(disease_names.keys())

print(f"\n✅ Loaded {len(disease_codes)} disease classes")
print(f"TV (Tortuous Vessels) is at index: {disease_codes.index('TV')}")

# Test with different inputs
test_cases = [
    ("Random noise", np.random.randn(1, 3, 224, 224).astype(np.float32)),
    ("All zeros", np.zeros((1, 3, 224, 224), dtype=np.float32)),
    ("All ones", np.ones((1, 3, 224, 224), dtype=np.float32)),
    ("Normalized random", np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5),
]

print(f"\n🧪 Testing model with different inputs:\n")

for test_name, test_input in test_cases:
    print(f"{'='*60}")
    print(f"Test: {test_name}")
    print(f"Input stats: min={test_input.min():.3f}, max={test_input.max():.3f}, mean={test_input.mean():.3f}")
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get predictions
    predictions = output[0]
    
    print(f"Output stats: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")
    print(f"Output sum: {predictions.sum():.6f}")
    print(f"Output std: {predictions.std():.6f}")
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions)[::-1][:5]
    
    print(f"\n🎯 Top 5 predictions:")
    for i, idx in enumerate(top_indices, 1):
        code = disease_codes[idx]
        name = disease_names[code]
        confidence = predictions[idx]
        print(f"  {i}. {name} ({code}): {confidence:.6f} ({confidence*100:.2f}%)")
    
    # Check if TV is in top predictions
    tv_index = disease_codes.index('TV')
    tv_confidence = predictions[tv_index]
    tv_rank = np.where(top_indices == tv_index)[0][0] + 1 if tv_index in top_indices else '>5'
    print(f"\n📍 TV (Tortuous Vessels) - Index {tv_index}: {tv_confidence:.6f} (Rank: {tv_rank})")
    print()

# Additional diagnostics
print(f"\n{'='*60}")
print(f"🔬 Additional Diagnostics:")
print(f"{'='*60}")

# Test if model outputs are always the same
input1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
input2 = np.random.randn(1, 3, 224, 224).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input1)
interpreter.invoke()
output1 = interpreter.get_tensor(output_details[0]['index'])[0]

interpreter.set_tensor(input_details[0]['index'], input2)
interpreter.invoke()
output2 = interpreter.get_tensor(output_details[0]['index'])[0]

difference = np.abs(output1 - output2)
print(f"\n📊 Output difference between two random inputs:")
print(f"  Max difference: {difference.max():.6f}")
print(f"  Mean difference: {difference.mean():.6f}")
print(f"  Are outputs identical? {np.allclose(output1, output2)}")

if np.allclose(output1, output2):
    print(f"\n⚠️  WARNING: Model produces identical outputs for different inputs!")
    print(f"   This suggests the model is not properly trained or has an issue.")
else:
    print(f"\n✅ Model produces different outputs for different inputs (expected behavior)")

# Check output distribution
print(f"\n📈 Output distribution for random input:")
interpreter.set_tensor(input_details[0]['index'], input1)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

print(f"  Min: {output.min():.6f}")
print(f"  Max: {output.max():.6f}")
print(f"  Mean: {output.mean():.6f}")
print(f"  Median: {np.median(output):.6f}")
print(f"  Std: {output.std():.6f}")
print(f"  Sum: {output.sum():.6f}")

# Check if outputs look like probabilities
is_probability_like = (output.min() >= 0) and (output.max() <= 1) and np.isclose(output.sum(), 1.0, atol=0.01)
print(f"\n  Looks like probabilities (0-1, sum≈1)? {is_probability_like}")

if not is_probability_like:
    print(f"  ⚠️  Outputs don't look like probabilities. May need softmax activation.")

# Check for always-high predictions at specific indices
print(f"\n📊 Checking for biased predictions:")
high_predictions = []
for idx, val in enumerate(output):
    if val > 0.1:  # Threshold for "high" prediction
        code = disease_codes[idx]
        name = disease_names[code]
        high_predictions.append((idx, code, name, val))

if high_predictions:
    print(f"  Classes with >10% confidence:")
    for idx, code, name, val in high_predictions:
        print(f"    {name} ({code}) [idx={idx}]: {val:.4f} ({val*100:.2f}%)")
else:
    print(f"  No classes have >10% confidence")

print(f"\n✅ Diagnostic complete!")
