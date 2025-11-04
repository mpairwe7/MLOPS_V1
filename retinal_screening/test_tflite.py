#!/usr/bin/env python3
"""
Test TFLite model inference
"""

import numpy as np
import tensorflow as tf
import sys

if len(sys.argv) < 2:
    print("Usage: python test_tflite.py <model.tflite>")
    sys.exit(1)

tflite_file = sys.argv[1]

print(f"Testing TFLite model: {tflite_file}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Create test input
test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"✓ Inference successful!")
print(f"Output shape: {output.shape}")
print(f"Output type: {output.dtype}")
print(f"Sample predictions: {output[0][:5]}")  # First 5 classes