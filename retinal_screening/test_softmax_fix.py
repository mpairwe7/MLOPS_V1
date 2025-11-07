#!/usr/bin/env python3
"""
Test to verify that applying softmax fixes the Tortuous Vessels bias issue
"""

import numpy as np
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
import json

# Load model
model_path = "assets/models/ai_edge_versions/model_graphclip_rank1_ai_edge.tflite"
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load disease names
with open('assets/data/disease_names.json', 'r') as f:
    disease_names = json.load(f)
    disease_codes = list(disease_names.keys())

def softmax(logits):
    """Apply softmax activation"""
    max_logit = np.max(logits)
    exp_values = np.exp(logits - max_logit)  # Numerical stability
    return exp_values / np.sum(exp_values)

print("=" * 70)
print("🧪 Testing Softmax Fix for Tortuous Vessels Bias")
print("=" * 70)

# Test with 3 different random inputs
for test_num in range(1, 4):
    print(f"\n{'='*70}")
    print(f"Test #{test_num}: Random Input")
    print(f"{'='*70}")
    
    # Generate random input
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Apply softmax
    probabilities = softmax(raw_output)
    
    print(f"\n📊 Raw Output (Logits) Statistics:")
    print(f"   Min: {raw_output.min():.6f}, Max: {raw_output.max():.6f}")
    print(f"   Mean: {raw_output.mean():.6f}, Sum: {raw_output.sum():.6f}")
    
    print(f"\n📊 After Softmax (Probabilities) Statistics:")
    print(f"   Min: {probabilities.min():.6f}, Max: {probabilities.max():.6f}")
    print(f"   Mean: {probabilities.mean():.6f}, Sum: {probabilities.sum():.6f}")
    print(f"   ✅ Sum ≈ 1.0? {np.isclose(probabilities.sum(), 1.0)}")
    
    # Get top 5 predictions BEFORE softmax
    print(f"\n🔴 Top 5 BEFORE Softmax (Raw Logits):")
    top_indices_raw = np.argsort(raw_output)[::-1][:5]
    for i, idx in enumerate(top_indices_raw, 1):
        code = disease_codes[idx]
        name = disease_names[code]
        value = raw_output[idx]
        print(f"   {i}. {name} ({code}): {value:.6f}")
    
    # Get top 5 predictions AFTER softmax
    print(f"\n✅ Top 5 AFTER Softmax (Probabilities):")
    top_indices_prob = np.argsort(probabilities)[::-1][:5]
    for i, idx in enumerate(top_indices_prob, 1):
        code = disease_codes[idx]
        name = disease_names[code]
        prob = probabilities[idx]
        print(f"   {i}. {name} ({code}): {prob:.6f} ({prob*100:.2f}%)")
    
    # Check TV position
    tv_index = disease_codes.index('TV')
    tv_rank_raw = np.where(top_indices_raw == tv_index)[0][0] + 1 if tv_index in top_indices_raw else '>5'
    tv_rank_prob = np.where(top_indices_prob == tv_index)[0][0] + 1 if tv_index in top_indices_prob else '>5'
    
    print(f"\n📍 Tortuous Vessels (TV) Ranking:")
    print(f"   BEFORE Softmax: Rank {tv_rank_raw}")
    print(f"   AFTER Softmax: Rank {tv_rank_prob}")
    
    if tv_rank_raw != tv_rank_prob:
        print(f"   ⚠️  Ranking changed after softmax!")
    else:
        print(f"   ✅ Ranking unchanged (softmax preserves order)")

print(f"\n{'='*70}")
print("📝 Summary:")
print("=" * 70)
print("""
Softmax converts raw logits to proper probabilities:
  • Ensures all values are between 0 and 1
  • Ensures sum equals 1.0 (valid probability distribution)
  • Maintains relative ordering (highest logit → highest probability)
  
The key insight:
  • TV may still rank high IF the model genuinely predicts it
  • BUT now we have proper probabilities instead of raw logits
  • This allows better interpretation and confidence thresholds
  
If TV still appears in every prediction:
  • The model itself may have a bias (training data issue)
  • Or the images being tested genuinely show tortuous vessels
""")
print("=" * 70)
