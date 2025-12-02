#!/usr/bin/env python3
"""Quick test to verify the filtered dataset loads correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_force_dummy import RobomimicForceDataset

hdf5_path = Path("~/multi-modal/data/robomimic/success_filtered_2025_11_11_delta_gripper_delta.hdf5").expanduser()

print("Creating dataset...")
dataset = RobomimicForceDataset(
    hdf5_path=hdf5_path,
    image_key_left="obs/wrist_image_left_rgb",
    image_key_right="obs/wrist_image_right_rgb",
    tactile_key="obs/tactile_values",
    gripper_key="obs/delta_gripper_position",  # Using pre-computed deltas
    tactile_length=50,
    tactile_channels=6,
    action_chunk_size=10,
    tactile_pad_value=0.0,
    tactile_window=50,
    image_size=224,
    normalize_images=True,
    # idle_epsilon=0.0,  # Not needed - data already filtered offline
)

print(f"\n✅ Dataset loaded successfully!")
print(f"  Total samples: {len(dataset)}")
# print(f"  Filtered idle: {dataset.filtered_idle}")  # Not tracked - data already filtered offline

print(f"\nTesting first sample...")
image_left, image_right, tactile, action_chunk = dataset[0]
print(f"  Image left shape: {image_left.shape}")
print(f"  Image right shape: {image_right.shape}")
print(f"  Tactile shape: {tactile.shape}")
print(f"  Action chunk shape: {action_chunk.shape}")
print(f"  Action chunk values: {action_chunk.numpy()}")

print(f"\n✅ All tests passed!")
