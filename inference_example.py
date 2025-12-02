#!/usr/bin/env python3
"""
Simple example showing how to load and use a trained model for inference.

This script demonstrates the minimal code needed to:
1. Load a trained checkpoint
2. Run inference on sample data
3. Get predictions
"""

import torch
from pathlib import Path
from model import MultimodalForceTransformer, MultimodalTransformerConfig


def simple_inference_example():
    """Minimal example of loading and using the model."""

    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create model with same config as training
    config = MultimodalTransformerConfig(
        image_encoder_type="dino_v3",
        dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        dinov3_freeze_backbone=True,
        dinov3_drop_cls_token=True,
        dinov3_normalize_inputs=True,
        tactile_channels=6,
        tactile_tokens=3,
        perceiver_num_latents=10,
        perceiver_num_layers=2,
        perceiver_num_heads=8,
        use_perceiver_resampler=True,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    )

    model = MultimodalForceTransformer(config)

    # 3. Load checkpoint
    checkpoint_path = Path("checkpoints/delta_gripper.pt")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"✅ Model loaded from {checkpoint_path}")

    # 4. Prepare sample data
    # Image: (batch_size, 3, height, width) - normalized to [0, 1]
    # Tactile: (batch_size, sequence_length, channels)
    batch_size = 1
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    tactile = torch.randn(batch_size, 50, 6).to(device)

    # 5. Run inference
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(image, tactile)

    # 6. Get result
    delta_gripper = prediction.squeeze().item()
    print(f"\nPredicted gripper delta: {delta_gripper:.6f}")

    return delta_gripper


def batch_inference_example():
    """Example of running inference on multiple samples."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (simplified - use config from training)
    config = MultimodalTransformerConfig()
    model = MultimodalForceTransformer(config)

    checkpoint_path = Path("checkpoints/delta_gripper.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Batch of samples
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    tactile = torch.randn(batch_size, 50, 6).to(device)

    # Run batch inference
    with torch.no_grad():
        predictions = model(images, tactile)

    # predictions shape: (batch_size, 1)
    predictions = predictions.squeeze(-1).cpu().numpy()

    print(f"\nBatch predictions ({batch_size} samples):")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: {pred:.6f}")

    return predictions


def inference_with_real_data():
    """Example using data from the dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_force_dummy import RobomimicForceDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    config = MultimodalTransformerConfig()
    model = MultimodalForceTransformer(config)
    checkpoint_path = Path("checkpoints/delta_gripper.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load dataset
    dataset = RobomimicForceDataset(
        hdf5_path=Path("~/multi-modal/data/robomimic/success_delta_2025_11_18_delta.hdf5").expanduser(),
        image_key="obs/wrist_image_left_rgb",
        tactile_key="obs/tactile_values",
        gripper_key="obs/delta_gripper_position",
        tactile_length=50,
        tactile_channels=6,
        tactile_pad_value=0.0,
        tactile_window=50,
        image_size=224,
        normalize_images=True,
    )

    # Get a sample
    image, tactile, target = dataset[0]

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    tactile = tactile.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(image, tactile)

    pred_value = prediction.item()
    target_value = target.item()

    print(f"\nReal data inference:")
    print(f"  Target:     {target_value:.6f}")
    print(f"  Prediction: {pred_value:.6f}")
    print(f"  Error:      {abs(pred_value - target_value):.6f}")

    return pred_value, target_value


if __name__ == "__main__":
    print("="*60)
    print("Example 1: Simple inference with random data")
    print("="*60)
    simple_inference_example()

    print("\n" + "="*60)
    print("Example 2: Batch inference")
    print("="*60)
    batch_inference_example()

    print("\n" + "="*60)
    print("Example 3: Inference with real dataset")
    print("="*60)
    try:
        inference_with_real_data()
    except FileNotFoundError as e:
        print(f"⚠️  Dataset not found: {e}")
        print("   Skipping real data example")
