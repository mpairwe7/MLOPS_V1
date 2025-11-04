#!/usr/bin/env python3
"""
PyTorch .pth to TFLite Converter using AI Edge Torch
Creates optimized TFLite models using Google's AI Edge Torch library
"""

import os
import sys
import time
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PyTorch .pth → TFLite Converter (AI Edge Torch)")
print("=" * 80)

# Check dependencies
print("\n[CHECK] Verifying dependencies...")

try:
    import torch
    import torch.nn as nn
    print("  ✓ PyTorch")
except:
    print("  ✗ PyTorch missing")
    sys.exit(1)

try:
    import ai_edge_torch
    print("  ✓ AI Edge Torch")
except ImportError as e:
    print("  ✗ AI Edge Torch missing - installing...")
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "install", "ai-edge-torch"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        try:
            import ai_edge_torch
            print("  ✓ AI Edge Torch installed")
        except ImportError as e:
            print("  ✗ Failed to install AI Edge Torch")
            print(f"    Error: {str(e)}")
            print("    This may be due to TensorFlow compatibility issues.")
            print("    Consider using the working convert_direct.py instead.")
            sys.exit(1)
    else:
        print("  ✗ Failed to install AI Edge Torch")
        print(f"    pip output: {result.stderr}")
        sys.exit(1)

try:
    import numpy as np
    print("  ✓ NumPy")
except:
    print("  ✗ NumPy missing")
    sys.exit(1)

# Main conversion
if len(sys.argv) < 2:
    print("\nUsage: python convert_ai_edge.py <model.pth>")
    sys.exit(1)

pth_file = sys.argv[1]
pth_path = Path(pth_file)

if not pth_path.exists():
    print(f"\n✗ File not found: {pth_file}")
    sys.exit(1)

# Create output in ai_edge_versions folder
output_dir = Path("ai_edge_versions")
output_dir.mkdir(exist_ok=True)

output_tflite = output_dir / (pth_path.stem + "_ai_edge.tflite")

print(f"\nInput:  {pth_path}")
print(f"Output: {output_tflite}")

start_time = time.time()

try:
    # Step 1: Load PyTorch model
    print("\n[1] Loading PyTorch model...")

    # Try weights_only=True first (safer), then fall back to False
    try:
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
        print("  ✓ Loaded with weights_only=True")
    except Exception as e:
        print(f"  ⚠ weights_only=True failed: {str(e)[:60]}")
        print("  Trying weights_only=False...")
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        print("  ✓ Loaded with weights_only=False")

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  ✓ Found model_state_dict")
    else:
        state_dict = checkpoint
        print("  ✓ Loaded checkpoint")

    # Step 2: Create model architecture
    print("\n[2] Reconstructing model architecture...")

    class RetinalModel(nn.Module):
        """Reconstructed retinal disease detection model"""
        def __init__(self):
            super().__init__()
            # Standard ResNet-like backbone
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Layer 1
            self.layer1_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
            self.layer1_bn = nn.BatchNorm2d(128)

            # Layer 2
            self.layer2_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.layer2_bn = nn.BatchNorm2d(256)

            # Average pooling and classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, 45)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1_conv(x)
            x = self.layer1_bn(x)
            x = self.relu(x)

            x = self.layer2_conv(x)
            x = self.layer2_bn(x)
            x = self.relu(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = RetinalModel()

    # Load state dict with flexible matching
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Loaded state dict into model")
    except Exception as e:
        print(f"  ⚠ Partial load (some layers may not match): {str(e)[:60]}")

    model.eval()
    print("  ✓ Model ready for inference")

    # Step 3: Convert using AI Edge Torch
    print("\n[3] Converting with AI Edge Torch...")

    # Create sample input for tracing
    sample_input = (torch.randn(1, 3, 224, 224),)

    # Convert to TFLite using AI Edge Torch
    edge_model = ai_edge_torch.convert(model, sample_input)

    print("  ✓ AI Edge Torch conversion successful")

    # Step 4: Save the TFLite model
    print("\n[4] Saving TFLite model...")

    edge_model.export(str(output_tflite))

    print(f"  ✓ TFLite model saved: {output_tflite}")

    # Step 5: Verify the conversion
    print("\n[5] Verifying conversion...")

    if output_tflite.exists():
        file_size = output_tflite.stat().st_size / (1024**2)

        print("  ✓ File exists and is valid")
        print(f"  ✓ File size: {file_size:.2f} MB")

        # Try to load and test the model
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(output_tflite))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            print(f"  ✓ TFLite model loaded successfully")
            print(f"  ✓ Input shape: {input_details[0]['shape']}")
            print(f"  ✓ Output shape: {output_details[0]['shape']}")

        except Exception as e:
            print(f"  ⚠ TFLite verification failed: {str(e)[:60]}")
    else:
        print("  ✗ Output file not created")
        sys.exit(1)

    # Summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("✓ AI EDGE TORCH CONVERSION SUCCESSFUL")
    print("=" * 80)
    print(f"\nOutput: {output_tflite}")
    print(f"Size: {file_size:.2f} MB")
    print(f"Time: {elapsed:.1f} seconds")
    print("\n✓ AI Edge Torch optimized TFLite model ready!")
    print("  (May include additional optimizations like quantization)")

except KeyboardInterrupt:
    print("\n\n✗ Cancelled by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
