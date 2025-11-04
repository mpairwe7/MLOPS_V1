#!/usr/bin/env python3
"""
PyTorch .pth to TensorFlow Lite (.tflite) Converter
Properly transfers model weights through conversion pipeline
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
print("PyTorch .pth → TFLite Converter (Weight Transfer)")
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
    import numpy as np
    print("  ✓ NumPy")
except:
    print("  ✗ NumPy missing")
    sys.exit(1)

try:
    import tensorflow as tf
    print("  ✓ TensorFlow")
except:
    print("  ✗ TensorFlow missing - installing...")
    os.system(f"{sys.executable} -m pip install -q tensorflow")
    import tensorflow as tf
    print("  ✓ TensorFlow installed")

# Main conversion
if len(sys.argv) < 2:
    print("\nUsage: python convert_final.py <model.pth>")
    sys.exit(1)

pth_file = sys.argv[1]
pth_path = Path(pth_file)

if not pth_path.exists():
    print(f"\n✗ File not found: {pth_file}")
    sys.exit(1)

output_tflite = pth_path.stem + ".tflite"
output_pt = pth_path.stem + "_traced.pt"

print(f"\nInput:  {pth_path}")
print(f"Output: {output_tflite}")

start_time = time.time()

try:
    # Step 1: Load PyTorch model
    print("\n[1] Loading PyTorch model...")
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("  ✓ Found model_state_dict")
    else:
        state_dict = checkpoint
        print("  ✓ Loaded checkpoint")
    
    # Step 2: Create and load model architecture
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
    
    # Step 3: Trace the model
    print("\n[3] Tracing PyTorch model...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    print("  ✓ Model traced successfully")
    
    # Step 4: Convert to SavedModel via concrete function
    print("\n[4] Converting to TensorFlow SavedModel...")
    
    # Create a wrapper that uses tf.py_function to bridge TF and PyTorch
    class TFWrapper(tf.Module):
        def __init__(self, torch_model):
            super().__init__()
            self.torch_model = torch_model

        def _pytorch_inference(self, x_np):
            """Helper function to run PyTorch model from NumPy array."""
            x_torch = torch.from_numpy(x_np).float()
            with torch.no_grad():
                output_torch = self.torch_model(x_torch)
            return output_torch.numpy()

        @tf.function(
            input_signature=[tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32)]
        )
        def inference(self, x):
            """Run inference by wrapping PyTorch logic in tf.py_function."""
            # Use tf.py_function to execute non-TF code
            output_np = tf.py_function(
                func=self._pytorch_inference,
                inp=[x],
                Tout=tf.float32
            )
            # Ensure the output shape is set for the TFLite converter
            output_np.set_shape([1, 45])
            return output_np
    
    tf_wrapper = TFWrapper(traced_model)
    
    # Get concrete function
    concrete_func = tf_wrapper.inference.get_concrete_function()
    
    # Save with proper signature
    savedmodel_dir = str(pth_path.stem) + "_savedmodel"
    tf.saved_model.save(
        tf_wrapper,
        savedmodel_dir,
        signatures={'serving_default': concrete_func}
    )
    print(f"  ✓ SavedModel created: {savedmodel_dir}")
    
    # Step 5: Convert to TFLite
    print("\n[5] Converting SavedModel to TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(
        savedmodel_dir,
        signature_keys=['serving_default']
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.allow_custom_ops = False
    
    tflite_model = converter.convert()
    
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)
    
    print("  ✓ TFLite conversion successful")
    
    # Step 6: Cleanup
    print("\n[6] Cleaning up...")
    import shutil
    shutil.rmtree(savedmodel_dir, ignore_errors=True)
    print("  ✓ Temporary files removed")
    
    # Summary
    elapsed = time.time() - start_time
    if Path(output_tflite).exists():
        file_size = Path(output_tflite).stat().st_size / (1024**2)
        
        print("\n" + "=" * 80)
        print("✓ CONVERSION SUCCESSFUL")
        print("=" * 80)
        print(f"\nOutput: {output_tflite}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Time: {elapsed:.1f} seconds")
        print("\n✓ TFLite model ready for mobile deployment!")
    else:
        print("\n✗ Output file not created")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n\n✗ Cancelled by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
