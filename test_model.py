#!/usr/bin/env python3
"""
Test/inference script for trained multimodal force prediction models.

Usage:
    # Test on validation dataset
    python test_model.py --checkpoint checkpoints/delta_gripper.pt

    # Test on dummy data
    python test_model.py --checkpoint checkpoints/delta_gripper.pt --use-dummy

    # Test single sample and visualize
    python test_model.py --checkpoint checkpoints/delta_gripper.pt --sample-index 0 --visualize

    # Run inference on custom data
    python test_model.py --checkpoint checkpoints/delta_gripper.pt --image path/to/image.jpg --tactile path/to/tactile.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import MultimodalForceTransformer, MultimodalTransformerConfig
from train_force_dummy import (
    DummyForceDataset,
    RobomimicForceDataset,
    RegressionMetricAccumulator,
    load_config,
)


def load_model(checkpoint_path: Path, config: dict) -> MultimodalForceTransformer:
    """Load model from checkpoint with configuration."""
    print(f"Loading model from {checkpoint_path}...")

    # Build model config from training config
    model_section = config.get("model", {})
    model_config = MultimodalTransformerConfig()

    # Override defaults with config values
    for key, value in model_section.items():
        if hasattr(model_config, key) and value is not None:
            setattr(model_config, key, value)

    # Instantiate model
    model = MultimodalForceTransformer(model_config)

    # Load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Config: {model_config.image_encoder_type}, "
          f"d_model={model_config.d_model}, "
          f"layers={model_config.num_layers}")

    return model

# why is this model: torch.nn.Module? i thought the model is checkpoints/delta_gripper.pt 
def test_on_dataset(model: torch.nn.Module, dataset, device: torch.device, batch_size: int = 16):
    """Run inference on entire dataset and compute metrics."""
    print(f"\nTesting on dataset with {len(dataset)} samples...")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics = RegressionMetricAccumulator()

    all_predictions = []
    all_targets = []

    with torch.no_grad(): # what does this mean? 
        for i, (images_left, images_right, tactile, target) in enumerate(loader): # what does this do? 
            images_left = images_left.to(device) # what does this do? 
            images_right = images_right.to(device)
            tactile = tactile.to(device)
            target = target.to(device).float()

            # Run inference
            pred = model(images_left, images_right, tactile) # action chunk output 

            # Accumulate metrics
            metrics.update(pred, target)
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"   Processed {(i + 1) * batch_size} samples...")

    # Compute final metrics
    results = metrics.results()

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Samples:        {int(results['count'])}")
    print(f"  MSE (Loss):     {results['mse']:.6f}")
    print(f"  MAE:            {results['mae']:.6f}")
    print(f"  RMSE:           {results['rmse']:.6f}")
    print(f"  R² Score:       {results['r2']:.6f}")
    print(f"  Pearson Corr:   {results['pearson']:.6f}")
    print(f"{'='*60}\n")

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return results, all_predictions, all_targets


def test_single_sample(
    model: torch.nn.Module,
    dataset,
    index: int,
    device: torch.device,
    visualize: bool = False
):
    """Test on a single sample and optionally visualize."""
    print(f"\nTesting single sample at index {index}...")

    if index >= len(dataset):
        raise IndexError(f"Index {index} out of range for dataset of size {len(dataset)}")

    # Get sample
    image_left, image_right, tactile, target = dataset[index]

    # Add batch dimension
    image_left = image_left.unsqueeze(0).to(device)
    image_right = image_right.unsqueeze(0).to(device)
    tactile = tactile.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(image_left, image_right, tactile)  # shape: (1, action_chunk_size)

    pred_chunk = prediction[0].cpu().numpy()
    target_chunk = target.numpy()
    error = abs(pred_value - target_value)

    print(f"\n{'='*60}")
    print(f"SINGLE SAMPLE TEST (Index {index})")
    print(f"{'='*60}")
    print(f"  Image left shape:  {tuple(image_left.shape)}")
    print(f"  Image right shape: {tuple(image_right.shape)}")
    print(f"  Tactile shape:     {tuple(tactile.shape)}")
    print(f"  Target chunk:      {target_chunk}")
    print(f"  Prediction chunk: {pred_chunk}")
    mse = np.mean((pred_chunk - target_chunk) ** 2)
    mae = np.mean(np.abs(pred_chunk - target_chunk))
    print(f"  MSE:               {mse:.6f}")
    print(f"  MAE:               {mae:.6f}")
    print(f"{'='*60}\n")

    if visualize:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Plot images
            img_left_np = image_left[0].cpu().numpy().transpose(1, 2, 0)
            img_right_np = image_right[0].cpu().numpy().transpose(1, 2, 0)
            if img_left_np.max() <= 1.0:
                img_left_np = np.clip(img_left_np, 0, 1)
            if img_right_np.max() <= 1.0:
                img_right_np = np.clip(img_right_np, 0, 1)
            axes[0].imshow(img_left_np)
            axes[0].set_title("Input Image Left")
            axes[0].axis("off")
            
            # Add second subplot for right image if needed
            if len(axes) < 3:
                fig.add_subplot(1, 3, 2)
                axes = fig.get_axes()
            axes[1].imshow(img_right_np)
            axes[1].set_title("Input Image Right")
            axes[1].axis("off")

            # Plot tactile data
            tactile_np = tactile[0].cpu().numpy()
            ax_idx = 2 if len(axes) > 2 else 1
            for i in range(min(6, tactile_np.shape[1])):
                axes[ax_idx].plot(tactile_np[:, i], label=f"Channel {i}")
            axes[ax_idx].set_title("Tactile Sequence")
            axes[ax_idx].set_xlabel("Time Step")
            axes[ax_idx].set_ylabel("Value")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

            plt.suptitle(f"Target MSE: {mse:.4f} | Prediction MAE: {mae:.4f}")
            plt.tight_layout()

            # Save figure
            output_path = Path(f"sample_{index}_visualization.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {output_path}")
            plt.show()

        except ImportError:
            print("⚠️  matplotlib not available for visualization")

    return pred_chunk, target_chunk


def test_custom_input(
    model: torch.nn.Module,
    image_left_path: Path,
    image_right_path: Path,
    tactile_path: Path,
    device: torch.device,
    config: dict
):
    """Test on custom images and tactile data."""
    print(f"\nTesting on custom inputs...")
    print(f"  Image left:  {image_left_path}")
    print(f"  Image right: {image_right_path}")
    print(f"  Tactile:     {tactile_path}")

    # Load images
    try:
        from PIL import Image
        img_left = Image.open(image_left_path).convert("RGB")
        img_left = np.array(img_left).transpose(2, 0, 1)  # HWC -> CHW
        img_left = torch.from_numpy(img_left).float() / 255.0
        
        img_right = Image.open(image_right_path).convert("RGB")
        img_right = np.array(img_right).transpose(2, 0, 1)  # HWC -> CHW
        img_right = torch.from_numpy(img_right).float() / 255.0
    except ImportError:
        raise ImportError("PIL/Pillow required for loading images. Install with: pip install Pillow")

    # Load tactile data (expected: numpy array of shape (50, 6))
    tactile = np.load(tactile_path)
    if tactile.ndim == 1:
        tactile = tactile.reshape(-1, 6)

    # Ensure correct shape
    tactile_length = config.get("robomimic", {}).get("tactile_length", 50)
    if tactile.shape[0] < tactile_length:
        # Pad
        pad = np.zeros((tactile_length - tactile.shape[0], tactile.shape[1]))
        tactile = np.concatenate([pad, tactile], axis=0)
    elif tactile.shape[0] > tactile_length:
        # Truncate
        tactile = tactile[-tactile_length:]

    tactile = torch.from_numpy(tactile).float()

    # Add batch dimension
    img_left = img_left.unsqueeze(0).to(device)
    img_right = img_right.unsqueeze(0).to(device)
    tactile = tactile.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        prediction = model(img_left, img_right, tactile)  # shape: (1, action_chunk_size)

    pred_chunk = prediction[0].cpu().numpy()

    print(f"\n{'='*60}")
    print(f"CUSTOM INPUT TEST")
    print(f"{'='*60}")
    print(f"  Prediction chunk: {pred_chunk}")
    print(f"{'='*60}\n")

    return pred_chunk


def main():
    parser = argparse.ArgumentParser(description="Test trained multimodal force prediction model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/delta_gripper.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to configs/default.yaml)"
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use dummy dataset instead of real data"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Test on a single sample at this index"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the sample (requires matplotlib)"
    )
    parser.add_argument(
        "--image-left",
        type=str,
        default=None,
        help="Path to custom left image file for inference"
    )
    parser.add_argument(
        "--image-right",
        type=str,
        default=None,
        help="Path to custom right image file for inference"
    )
    parser.add_argument(
        "--tactile",
        type=str,
        default=None,
        help="Path to custom tactile data (.npy) for inference"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for dataset testing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect by default)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint).expanduser()
    model = load_model(checkpoint_path, config)
    model = model.to(device)

    # Test on custom input
    if args.image_left and args.image_right and args.tactile:
        test_custom_input(
            model,
            Path(args.image_left).expanduser(),
            Path(args.image_right).expanduser(),
            Path(args.tactile).expanduser(),
            device,
            config
        )
        return

    # Build dataset
    if args.use_dummy:
        print("\nUsing dummy validation dataset...")
        dataset = DummyForceDataset(
            split="val",
            length=config.get("dummy_val_size", 128),
            image_size=config.get("dummy_image_size", 224),
            tactile_len=config.get("dummy_tactile_length", 50),
            action_chunk_size=config.get("action_chunk_size", 10),
            seed=config.get("dummy_seed", 0),
        )
    else:
        print("\nLoading validation dataset from config...")
        robomimic_cfg = config.get("robomimic", {})
        val_path = robomimic_cfg.get("val_path")

        if not val_path:
            print("No validation path found in config. Using dummy data.")
            dataset = DummyForceDataset(
                split="val",
                length=config.get("dummy_val_size", 128),
                image_size=config.get("dummy_image_size", 224),
                tactile_len=config.get("dummy_tactile_length", 50),
                action_chunk_size=config.get("action_chunk_size", 10),
                seed=config.get("dummy_seed", 0),
            )
        else:
            dataset = RobomimicForceDataset(
                hdf5_path=val_path,
                image_key_left=robomimic_cfg["image_key_left"],
                image_key_right=robomimic_cfg["image_key_right"],
                tactile_key=robomimic_cfg["tactile_key"],
                gripper_key=robomimic_cfg["gripper_key"],
                tactile_length=robomimic_cfg.get("tactile_length", 50),
                tactile_channels=robomimic_cfg.get("tactile_channels", 6),
                action_chunk_size=robomimic_cfg.get("action_chunk_size", config.get("action_chunk_size", 10)),
                tactile_pad_value=robomimic_cfg.get("tactile_pad_value", 0.0),
                tactile_window=robomimic_cfg.get("tactile_window"),
                image_size=robomimic_cfg.get("image_size"),
                normalize_images=robomimic_cfg.get("normalize_images", True),
            )

    # Test single sample
    if args.sample_index is not None:
        test_single_sample(model, dataset, args.sample_index, device, args.visualize)
    else:
        # Test on full dataset
        test_on_dataset(model, dataset, device, args.batch_size)


if __name__ == "__main__":
    main()
