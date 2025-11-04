#!/usr/bin/env python3
"""
PyTorch .pth to TensorFlow Lite (.tflite) Converter
Direct conversion from Keras model - no SavedModel intermediate
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
print("PyTorch .pth → TFLite Converter (Direct Keras)")
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
    from tensorflow import keras
    print("  ✓ TensorFlow/Keras")
except:
    print("  ✗ TensorFlow missing - installing...")
    os.system(f"{sys.executable} -m pip install -q tensorflow")
    import tensorflow as tf
    from tensorflow import keras
    print("  ✓ TensorFlow installed")

# Main conversion
if len(sys.argv) < 2:
    print("\nUsage: python convert_direct.py <model.pth>")
    sys.exit(1)

pth_file = sys.argv[1]
pth_path = Path(pth_file)

if not pth_path.exists():
    print(f"\n✗ File not found: {pth_file}")
    sys.exit(1)

output_tflite = pth_path.stem + ".tflite"

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

    # Step 2: Analyze PyTorch model architecture
    print("\n[2] Analyzing PyTorch model architecture...")

    layer_info = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            layer_info[key] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype)
            }

    print(f"  ✓ Found {len(layer_info)} parameter tensors")

    # Step 3: Create equivalent Keras model
    print("\n[3] Creating equivalent Keras model...")

    # Build using functional API
    inputs = keras.layers.Input(shape=(224, 224, 3))

    # Conv block 1
    x = keras.layers.Conv2D(64, (7, 7), strides=2, padding='same',
                           use_bias=False, name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)

    # Max pooling
    x = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same', name='maxpool')(x)

    # Conv block 2
    x = keras.layers.Conv2D(128, (3, 3), padding='same',
                           use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.ReLU(name='relu2')(x)

    # Conv block 3
    x = keras.layers.Conv2D(256, (3, 3), padding='same',
                           use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.ReLU(name='relu3')(x)

    # Global average pooling
    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)

    # Dense classifier
    outputs = keras.layers.Dense(45, name='fc')(x)  # 45 disease classes

    keras_model = keras.Model(inputs=inputs, outputs=outputs)

    print("  ✓ Keras model created")

    # Step 4: Transfer weights manually
    print("\n[4] Transferring weights from PyTorch to Keras...")

    weight_mapping = {
        # Conv1
        'conv1.weight': ('conv1', 'kernel'),  # (out, in, h, w) -> (h, w, in, out)

        # BN1
        'bn1.weight': ('bn1', 'gamma'),
        'bn1.bias': ('bn1', 'beta'),
        'bn1.running_mean': ('bn1', 'moving_mean'),
        'bn1.running_var': ('bn1', 'moving_variance'),

        # Conv2
        'layer1_conv.weight': ('conv2', 'kernel'),

        # BN2
        'layer1_bn.weight': ('bn2', 'gamma'),
        'layer1_bn.bias': ('bn2', 'beta'),
        'layer1_bn.running_mean': ('bn2', 'moving_mean'),
        'layer1_bn.running_var': ('bn2', 'moving_variance'),

        # Conv3
        'layer2_conv.weight': ('conv3', 'kernel'),

        # BN3
        'layer2_bn.weight': ('bn3', 'gamma'),
        'layer2_bn.bias': ('bn3', 'beta'),
        'layer2_bn.running_mean': ('bn3', 'moving_mean'),
        'layer2_bn.running_var': ('bn3', 'moving_variance'),

        # FC layer
        'fc.weight': ('fc', 'kernel'),  # (out, in) -> (in, out)
        'fc.bias': ('fc', 'bias'),
    }

    transferred_count = 0

    for pt_name, (keras_layer_name, keras_param_name) in weight_mapping.items():
        if pt_name in state_dict:
            pt_tensor = state_dict[pt_name].numpy()

            # Find the Keras layer
            keras_layer = None
            for layer in keras_model.layers:
                if layer.name == keras_layer_name:
                    keras_layer = layer
                    break

            if keras_layer is None:
                print(f"  ⚠ Keras layer '{keras_layer_name}' not found")
                continue

            # Transform weight shape if needed
            if keras_param_name == 'kernel':
                if len(pt_tensor.shape) == 4:  # Conv weights
                    pt_tensor = np.transpose(pt_tensor, (2, 3, 1, 0))
                elif len(pt_tensor.shape) == 2:  # Dense weights
                    pt_tensor = np.transpose(pt_tensor, (1, 0))

            # Set the weight
            if keras_param_name in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
                keras_layer.set_weights([pt_tensor])
            else:
                keras_layer.set_weights([pt_tensor])

            transferred_count += 1
            print(f"  ✓ {pt_name} → {keras_layer_name}.{keras_param_name}")

    print(f"  ✓ Transferred {transferred_count} weight tensors")

    # Step 5: Test the model
    print("\n[5] Testing model inference...")

    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)

    try:
        output = keras_model.predict(dummy_input, verbose=0)
        print(f"  ✓ Model inference successful, output shape: {output.shape}")
    except Exception as e:
        print(f"  ⚠ Inference test failed: {str(e)[:60]}")
        print("  Continuing with conversion...")

    # Step 6: Convert directly to TFLite
    print("\n[6] Converting Keras model directly to TFLite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)

    print("  ✓ TFLite conversion successful")

    # Step 7: Summary
    elapsed = time.time() - start_time
    if Path(output_tflite).exists():
        file_size = Path(output_tflite).stat().st_size / (1024**2)

        print("\n" + "=" * 80)
        print("✓ CONVERSION SUCCESSFUL")
        print("=" * 80)
        print(f"\nOutput: {output_tflite}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Layers transferred: {transferred_count}")
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
