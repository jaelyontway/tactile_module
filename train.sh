#!/bin/bash
# Training script for multimodal force transformer with tactile and gripper control

set -e  # Exit on error

# Configuration file
CONFIG_FILE="${1:-configs/default.yaml}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/default.yaml"
    exit 1
fi

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="
echo ""

# Run training
python train_tactile_gripper.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
