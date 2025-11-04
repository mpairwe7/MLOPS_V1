#!/bin/bash
# Batch convert all .pth models using AI Edge Torch

echo "========================================="
echo "AI Edge Torch Batch Conversion"
echo "========================================="

# Activate environment
source myenv/bin/activate

# Navigate to models directory
cd assets/models

echo "Installing AI Edge Torch..."
pip install ai-edge-torch

echo ""
echo "Converting models..."

# Convert each .pth file
for pth_file in *.pth; do
    if [[ -f "$pth_file" ]]; then
        echo "----------------------------------------"
        echo "Converting: $pth_file"
        echo "----------------------------------------"

        python ../../convert_ai_edge.py "$pth_file"

        echo ""
    fi
done

echo "========================================="
echo "Conversion complete!"
echo "Check ai_edge_versions/ folder for results"
echo "========================================="