# Tactile Module

Multimodal PyTorch components for predicting grasping force from synchronized vision and tactile sensor traces. The package ships a reusable transformer architecture and a self-contained training script that can run either on synthetic dummy data or on a custom dataset that matches the expected interface.

## Features
- Image encoder that converts RGB frames into a grid of tokens compatible with transformer models.
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
