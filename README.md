# Tactile Module

Multimodal PyTorch components for predicting gripper action chunks from synchronized dual-camera vision and tactile sensor traces. The package ships a reusable transformer architecture and a self-contained training script that can run either on synthetic dummy data or on a custom dataset that matches the expected interface.

## Features
- **Dual-camera input**: Processes both left and right wrist camera images for richer visual context
- **Action chunking**: Predicts multiple future gripper deltas (default: 10 steps) instead of a single value
- Pretrained DINOv3 image encoder (via Hugging Face Transformers) that emits ViT patch tokens ready for fusion
- Tactile encoder that summarizes temporal sensor sequences (default: 50×6) into the same embedding space
- Transformer fusion block with a regression head that predicts action chunks
- Training harness (`train_force_dummy.py`) that demonstrates data loading, logging with Weights & Biases (optional), and checkpointing

## Installation
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project only depends on PyTorch and helper libraries listed in `requirements.txt`. Adjust the CUDA build as needed for your hardware.

## Quick Start
Run the reference training loop on the synthetic dataset:
```bash
python -m tactile_module.train_force_dummy
```
The script will:
1. Instantiate `MultimodalForceTransformer`.
2. Generate dummy RGB/tactile samples unless a custom `ForceDataset` implementation is available on the Python path.
3. Log metrics locally (Weights & Biases offline mode by default).
4. Save checkpoints under `checkpoints/multimodal_force.pt` when the directory is writable.

### Custom Dataset Integration
Provide a module named `my_dataset.py` with a `ForceDataset` class that matches the PyTorch `Dataset` interface and returns `(image_left, image_right, tactile, action_chunk)` tuples. When present, the training script will automatically prefer it over the dummy data:
```python
from torch.utils.data import Dataset

class ForceDataset(Dataset):
    def __getitem__(self, idx):
        # return image_left (3, H, W), image_right (3, H, W), 
        # tactile (50, 6), action_chunk (action_chunk_size,)
        ...
```

Run the training script with `use_dummy_data=False` in the config to switch to the real dataset.

### Using The DINOv3 Image Encoder
The default configuration now loads a pretrained DINOv3 transformer through the Hugging Face `transformers` package. Make sure `pip install -r requirements.txt` has been run so the dependency (and its `safetensors` helper) are available. Key knobs live in `MultimodalTransformerConfig` (`tactile_module/model.py`):

- `image_encoder_type`: set to `dino_v3` (default) to activate the pretrained backbone or `conv` to fall back to the lightweight CNN.
- `dinov3_model_name`: Hugging Face identifier for the checkpoint, e.g. `facebook/dinov3-base` or any custom repo with compatible weights.
- `dinov3_freeze_backbone`: freeze ViT weights during training (default `true`).
- `dinov3_drop_cls_token`: remove the CLS token so only patch tokens feed the multimodal transformer.

Inputs are automatically resized to the backbone’s advertised resolution (usually 224) and normalised with ImageNet statistics before being forwarded through DINOv3.

## Configuration
- Edit `configs/default.yaml` to change hyperparameters, logging options, or dataset settings. For example, updating `wandb_experiment` or `batch_size` in the YAML file automatically applies to the next training run.
- Override the path at runtime with `python -m tactile_module.train_force_dummy --config path/to/custom.yaml`.
- Use `scripts/train.sh` to launch training without worrying about `PYTHONPATH`; the script forwards any additional CLI arguments to the module entry point.
- To train on a Robomimic dataset, set `dataset_type: robomimic` and point `robomimic.train_path` / `robomimic.val_path` to the respective `.hdf5` files. Explicitly specify the observation keys (`image_key_left`, `image_key_right`, `tactile_key`, `gripper_key`) so the loader knows which streams to consume.
- The default configuration targets `~/multi-modal/data/robomimic/success_delta_2025_11_18_delta.hdf5` and uses both wrist cameras (`obs/wrist_image_left_rgb`, `obs/wrist_image_right_rgb`), tactile traces (`obs/tactile_values`), and gripper deltas (`obs/delta_gripper_position`). The model predicts action chunks of size 10 (configurable via `action_chunk_size`).
- Logged training metrics:
  - `train/grad_clip_rate`: fraction of updates that triggered gradient clipping; spikes flag potential gradient explosions.
  - `train/grad_norm_mean` / `train/grad_norm_max`: average and worst gradient norm per epoch; track stability and signal strength.
  - `train/mae` / `val/mae`: mean absolute error (in force units), an interpretable average deviation.
  - `train/rmse` / `val/rmse`: root mean square error, more sensitive to large mistakes than MAE.
  - `train/r2` / `val/r2`: coefficient of determination; 1.0 indicates perfect fit, 0 matches a constant baseline, negative is worse than baseline.
  - `train/pearson` / `val/pearson`: Pearson correlation between predictions and ground truth, highlighting whether trends align.

## Model Usage
```python
import torch
from tactile_module.model import MultimodalForceTransformer, MultimodalTransformerConfig

config = MultimodalTransformerConfig(action_chunk_size=10)
model = MultimodalForceTransformer(config)

images_left = torch.randn(4, 3, 224, 224)   # batch, channel, height, width
images_right = torch.randn(4, 3, 224, 224)  # batch, channel, height, width
tactile = torch.randn(4, 50, 6)              # batch, sequence length, channels
action_chunk = model(images_left, images_right, tactile)  # -> (4, 10)
```

The model processes both left and right wrist camera images separately, then concatenates their tokens before fusion with tactile features. The output is an action chunk predicting gripper deltas for the next `action_chunk_size` steps (default: 10).

### Real Robot (pi0) Gripper Override
Use the tactile model to replace the gripper dimension of pi0’s action chunk during rollout:
```python
from tactile_module.robot_inference_adapter import TactileGripperAdapter

adapter = TactileGripperAdapter(
    checkpoint_path="/home/pi0/multi-modal/tactile_module/checkpoints/delta_gripper.pt",
    config_path="/home/pi0/multi-modal/tactile_module/configs/default.yaml",
)

# Inside the droid control loop (see droid-multi-modal/scripts/1-pi0.py)
wrist_rgb_left = curr_obs["wrist_image_left"]       # H×W×3 RGB
wrist_rgb_right = curr_obs["wrist_image_right"]    # H×W×3 RGB
tactile_history = tactile_reader.read_values()      # (T, 6) from the ring buffer
pi0_action = pred_action_chunk[actions_from_chunk_completed]
current_gripper = float(curr_obs["gripper_position"][0])

# Option 1: Use single delta from action chunk (backward compatible)
merged_action, tactile_delta = adapter.override_gripper(
    pi0_action,
    wrist_rgb_left,
    wrist_rgb_right,
    tactile_history,
    current_gripper,
    pi0_gate=0.2,              # only override when pi0 is actively moving gripper
    absolute_clip=(-1.0, 1.0), # RobotEnv with gripper_action_space="position"
    step_index=0,              # use first step from action chunk
)
env.step(merged_action)

# Option 2: Get full action chunk for planning
action_chunk = adapter.predict_action_chunk(
    wrist_rgb_left,
    wrist_rgb_right,
    tactile_history
)  # -> (action_chunk_size,) array
```
This mirrors the training preprocessing: both wrist images are resized/normalised to 224×224 and tactile traces are padded/trimmed to the last 50×6 samples before the transformer predicts an action chunk of gripper deltas.

## Repository Structure
- `model.py`: Multimodal transformer implementation with dual-camera support and action chunking
- `train_force_dummy.py`: Reference training loop with dummy data and optional Weights & Biases logging
- `robot_inference_adapter.py`: Lightweight wrapper for real-robot inference with action chunk support
- `test_model.py`: Model testing and evaluation script
- `test_dataset.py`: Dataset loading verification script
- `configs/default.yaml`: Default training configuration
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment alternative

## Key Configuration Parameters
- `action_chunk_size`: Number of future gripper deltas to predict (default: 10)
- `image_key_left` / `image_key_right`: HDF5 keys for left and right wrist camera images
- `tactile_length`: Length of tactile sequence after padding/truncation (default: 50)
- `tactile_channels`: Number of tactile sensor channels (default: 6)

## License
Add your preferred license here before distributing the repository publicly.
