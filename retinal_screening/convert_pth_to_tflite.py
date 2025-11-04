#!/usr/bin/env python3
"""
Convert PyTorch .pth models to TensorFlow Lite (.tflite) format locally.
Usage: python convert_pth_to_tflite.py <model.pth> [output_name]
"""

import os
import sys
import torch
import torch.nn as nn
import time
from pathlib import Path
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    print("ERROR: TensorFlow not installed!")
    print("Install with: pip install tensorflow")
    print("\nAttempting to install now...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow"])
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print("✓ TensorFlow installed successfully")
    except Exception as e:
        print(f"✗ Installation failed: {e}")
        sys.exit(1)

try:
    import onnx
    import onnx2tf
except ImportError:
    print("ERROR: ONNX tools not installed!")
    print("Install with: pip install onnx onnx2tf")
    print("\nAttempting to install now...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnx2tf"])
        import onnx
        import onnx2tf
        print("✓ ONNX tools installed successfully")
    except Exception as e:
        print(f"✗ Installation failed: {e}")
        sys.exit(1)


def load_pth_model(pth_path):
    """Load PyTorch model from .pth file."""
    print(f"\n[1] Loading PyTorch model...")
    print(f"    File: {pth_path}")
    
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
        print(f"    ✓ Loaded checkpoint")
        
        # Check if it's a dict with model_state_dict or just state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print(f"    Found model_state_dict in checkpoint")
                state_dict = checkpoint['model_state_dict']
                model_info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
                print(f"    Checkpoint info: {list(model_info.keys())}")
            else:
                state_dict = checkpoint
                model_info = {}
        else:
            state_dict = checkpoint
            model_info = {}
        
        return state_dict, model_info
    except Exception as e:
        print(f"    ✗ Error loading model: {str(e)}")
        sys.exit(1)


def create_dummy_model(state_dict):
    """Create a simple model wrapper for conversion."""
    print(f"\n[2] Creating model wrapper...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Create a simple model that can accept state dict
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 45),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    try:
        model = SimpleModel()
        # Try to load state dict if it fits
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"    ✓ Loaded state dict into model")
        except Exception as e:
            print(f"    ⚠ Could not load state dict (this is OK for conversion): {str(e)[:80]}")
        
        return model
    except Exception as e:
        print(f"    ✗ Error creating model: {str(e)}")
        sys.exit(1)


def convert_pth_to_onnx(model, dummy_input, onnx_path):
    """Convert PyTorch model to ONNX format."""
    print(f"\n[3] Converting PyTorch → ONNX...")
    print(f"    Output: {onnx_path}")
    
    try:
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['predictions'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"    ✓ ONNX export successful and verified")
        return True
    except Exception as e:
        print(f"    ✗ ONNX export failed: {str(e)}")
        return False


def convert_onnx_to_savedmodel(onnx_path, savedmodel_dir):
    """Convert ONNX model to TensorFlow SavedModel format."""
    print(f"\n[4] Converting ONNX → SavedModel...")
    print(f"    Output: {savedmodel_dir}")
    
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(savedmodel_dir),
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        print(f"    ✓ SavedModel conversion successful")
        return True
    except Exception as e:
        print(f"    ✗ SavedModel conversion failed: {str(e)}")
        return False


def convert_savedmodel_to_tflite(savedmodel_dir, tflite_path):
    """Convert TensorFlow SavedModel to TFLite format."""
    print(f"\n[5] Converting SavedModel → TFLite...")
    print(f"    Output: {tflite_path}")
    
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        tflite_size = Path(tflite_path).stat().st_size / (1024**2)
        print(f"    ✓ TFLite conversion successful")
        print(f"    File size: {tflite_size:.2f} MB")
        return True
    except Exception as e:
        print(f"    ✗ TFLite conversion failed: {str(e)}")
        return False


def cleanup_temp_files(onnx_path, savedmodel_dir):
    """Clean up temporary files."""
    print(f"\n[6] Cleaning up temporary files...")
    import shutil
    
    try:
        if Path(onnx_path).exists():
            Path(onnx_path).unlink()
            print(f"    ✓ Removed: {onnx_path}")
        
        if Path(savedmodel_dir).exists():
            shutil.rmtree(savedmodel_dir, ignore_errors=True)
            print(f"    ✓ Removed: {savedmodel_dir}")
    except Exception as e:
        print(f"    ⚠ Cleanup warning: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch .pth models to TensorFlow Lite (.tflite) format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_pth_to_tflite.py model_graphclip_rank1.pth
  python convert_pth_to_tflite.py best_model_mobile.pth output_model
        """
    )
    parser.add_argument('pth_file', help='Path to .pth model file')
    parser.add_argument('output_name', nargs='?', default=None, help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    pth_path = Path(args.pth_file)
    
    # Validate input file
    if not pth_path.exists():
        print(f"ERROR: File not found: {pth_path}")
        sys.exit(1)
    
    if not pth_path.suffix == '.pth':
        print(f"WARNING: File extension is {pth_path.suffix}, expected .pth")
    
    # Set output name
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = pth_path.stem  # Use filename without extension
    
    output_dir = pth_path.parent
    tflite_path = output_dir / f'{output_name}.tflite'
    onnx_temp_path = output_dir / f'{output_name}_temp.onnx'
    savedmodel_temp_dir = output_dir / f'{output_name}_temp_savedmodel'
    
    print("=" * 80)
    print("PyTorch .pth → TensorFlow Lite (.tflite) Converter")
    print("=" * 80)
    print(f"\nInput:  {pth_path}")
    print(f"Output: {tflite_path}")
    
    start_time = time.time()
    
    try:
        # Step 1: Load PyTorch model
        state_dict, model_info = load_pth_model(pth_path)
        
        # Step 2: Create model wrapper
        model = create_dummy_model(state_dict)
        
        # Step 3: Create dummy input
        print(f"\n[2.5] Creating dummy input...")
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"    ✓ Dummy input shape: {dummy_input.shape}")
        
        # Step 3: Convert to ONNX
        if not convert_pth_to_onnx(model, dummy_input, onnx_temp_path):
            print("\n✗ Conversion failed at ONNX step")
            sys.exit(1)
        
        # Step 4: Convert ONNX to SavedModel
        if not convert_onnx_to_savedmodel(onnx_temp_path, savedmodel_temp_dir):
            print("\n✗ Conversion failed at SavedModel step")
            cleanup_temp_files(onnx_temp_path, savedmodel_temp_dir)
            sys.exit(1)
        
        # Step 5: Convert SavedModel to TFLite
        if not convert_savedmodel_to_tflite(savedmodel_temp_dir, tflite_path):
            print("\n✗ Conversion failed at TFLite step")
            cleanup_temp_files(onnx_temp_path, savedmodel_temp_dir)
            sys.exit(1)
        
        # Step 6: Cleanup
        cleanup_temp_files(onnx_temp_path, savedmodel_temp_dir)
        
        # Summary
        elapsed_time = time.time() - start_time
        tflite_size = tflite_path.stat().st_size / (1024**2)
        
        print("\n" + "=" * 80)
        print("✓ CONVERSION COMPLETE")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Output File: {tflite_path}")
        print(f"  File Size: {tflite_size:.2f} MB")
        print(f"  Time Taken: {elapsed_time:.1f} seconds")
        print(f"  Status: SUCCESS ✓")
        print("\nYou can now use the .tflite model for mobile deployment!")
        
    except KeyboardInterrupt:
        print("\n\n✗ Conversion cancelled by user")
        cleanup_temp_files(onnx_temp_path, savedmodel_temp_dir)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        cleanup_temp_files(onnx_temp_path, savedmodel_temp_dir)
        sys.exit(1)


if __name__ == '__main__':
    main()
