# Tactile Module

Multimodal PyTorch components for predicting grasping force from synchronized vision and tactile sensor traces. The package ships a reusable transformer architecture and a self-contained training script that can run either on synthetic dummy data or on a custom dataset that matches the expected interface.

## Features
- Pretrained DINOv3 image encoder (via Hugging Face Transformers) that emits ViT patch tokens ready for fusion.
- Tactile encoder that summarizes 500×6 sensor sequences into the same embedding space.
- Transformer fusion block with a regression head that predicts a single force value.
- Training harness (`train_force_dummy.py`) that demonstrates data loading, logging with Weights & Biases (optional), and checkpointing.

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
Provide a module named `my_dataset.py` with a `ForceDataset` class that matches the PyTorch `Dataset` interface and returns `(image, tactile, force)` tuples. When present, the training script will automatically prefer it over the dummy data:
```python
from torch.utils.data import Dataset

class ForceDataset(Dataset):
    def __getitem__(self, idx):
        # return image (3, H, W), tactile (500, 6), force scalar
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
- To train on a Robomimic dataset, set `dataset_type: robomimic` and point `robomimic.train_path` / `robomimic.val_path` to the respective `.hdf5` files. Explicitly specify the observation keys (`image_key`, `tactile_key`, `force_key`) so the loader knows which streams to consume.
- The default configuration targets `~/multi-modal/data/robomimic/success_2025_11_04.hdf5` and uses the wrist camera (`obs/wrist_image_left_rgb`), tactile traces (`obs/tactile_values`), and force predictions (`obs/force_prediction`). Update these entries if you swap in a dataset with different naming.
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

config = MultimodalTransformerConfig()
model = MultimodalForceTransformer(config)

images = torch.randn(4, 3, 224, 224)      # batch, channel, height, width
tactile = torch.randn(4, 500, 6)          # batch, sequence length, channels
force = model(images, tactile)            # -> (4, 1)
```

## Repository Structure
- `model.py`: Multimodal transformer implementation.
- `train_force_dummy.py`: Reference training loop with dummy data and optional Weights & Biases logging.
- `requirements.txt`: Python dependencies.
- `environment.yml`: Conda environment alternative.

## License
Add your preferred license here before distributing the repository publicly.
