import argparse
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from .model import MultimodalForceTransformer, MultimodalTransformerConfig

try:
    from my_dataset import ForceDataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ForceDataset = None


class DummyForceDataset(Dataset):
    """
    Synthetic dataset that mimics the real data interface.

    Each item returns:
        image: (3, image_size, image_size) float tensor
        tactile: (tactile_len, 6) float tensor
        force: scalar tensor correlated with the inputs
    """

    def __init__(
        self,
        split: str,
        length: int = 128,
        image_size: int = 224,
        tactile_len: int = 500,
        seed: int = 0,
    ):
        self.split = split
        self.length = length
        self.image_size = image_size
        self.tactile_len = tactile_len
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def _make_generator(self, idx: int) -> torch.Generator:
        gen = torch.Generator()
        base = 17 if self.split == "train" else 79
        gen.manual_seed(self.seed + base * (idx + 1))
        return gen

    def __getitem__(self, idx: int):
        gen = self._make_generator(idx)
        image = torch.randn(3, self.image_size, self.image_size, generator=gen)
        tactile = torch.randn(self.tactile_len, 6, generator=gen)

        # Simple synthetic target correlated with both modalities
        force = 5.0 * image.mean() + 2.0 * tactile.mean()
        force = force.clamp(min=0.0)

        return image.float(), tactile.float(), force.float()


def _ensure_tmpdir() -> Optional[Path]:
    candidates = []
    env_tmp = os.environ.get("TMPDIR")
    if env_tmp:
        candidates.append(Path(env_tmp))

    module_tmp = Path(__file__).resolve().parent / ".tmp"
    home_tmp = Path.home() / ".tmp"
    candidates.extend(
        [
            module_tmp,
            home_tmp,
            Path("/tmp"),
            Path("/var/tmp"),
            Path("/usr/tmp"),
        ]
    )

    for path in candidates:
        try:
            if path.exists():
                if path.is_dir() and os.access(str(path), os.W_OK):
                    os.environ["TMPDIR"] = str(path)
                    return path
                continue
            path.mkdir(parents=True, exist_ok=True)
            if os.access(str(path), os.W_OK):
                os.environ["TMPDIR"] = str(path)
                return path
        except (OSError, PermissionError):
            continue
    return None


def _initialize_wandb(config) -> Optional[Any]:
    if not config.get("use_wandb", True):
        return None

    _ensure_tmpdir()
    wandb_mode = config.get("wandb_mode")
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode
    else:
        os.environ.setdefault("WANDB_MODE", "offline")

    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"wandb unavailable ({exc}); continuing without logging.")
        return None

    try:
        project_name = config.get("wandb_project")
        experiment_name = config.get("wandb_experiment") or config.get("run_name")
        if project_name is None:
            raise KeyError("Missing 'wandb_project' in configuration.")
        wandb.init(project=project_name, config=config, name=experiment_name)
    except Exception as exc:  # pragma: no cover - wandb optional
        print(f"Failed to initialize wandb ({exc}); continuing without logging.")
        return None
    return wandb


def load_config(path: Optional[str] = None) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path: Optional path supplied by the caller. When absent, defaults to configs/default.yaml.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency expected via requirements
        raise RuntimeError(
            "PyYAML is required to load configuration files. Install it via `pip install pyyaml`."
        ) from exc

    default_path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    config_path = Path(path).expanduser() if path else default_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Configuration file '{config_path}' must define a top-level mapping.")

    float_keys = {"lr", "weight_decay", "dropout"}
    for key in float_keys:
        if key in data:
            try:
                data[key] = float(data[key])
            except (TypeError, ValueError):
                raise ValueError(f"Configuration value for '{key}' must be numeric, got {data[key]!r}.") from None

    return data


def train(config):
    wandb_module = _initialize_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalForceTransformer(MultimodalTransformerConfig()).to(device)
    if wandb_module is not None:
        wandb_module.watch(model, log_freq=100, log="all")

    def build_dataset(split: str):
        use_dummy = config.get("use_dummy_data", False) or ForceDataset is None
        if not use_dummy:
            if ForceDataset is None:
                raise RuntimeError("ForceDataset is unavailable. Enable 'use_dummy_data' or provide an implementation.")
            return ForceDataset(split=split)

        length = config["dummy_train_size"] if split == "train" else config["dummy_val_size"]
        return DummyForceDataset(
            split=split,
            length=length,
            image_size=config["dummy_image_size"],
            tactile_len=config["dummy_tactile_length"],
            seed=config.get("dummy_seed", 0),
        )

    train_loader = DataLoader(
        build_dataset("train"),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        build_dataset("val"),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    checkpoint_path = Path(config["checkpoint_path"]).expanduser()
    can_save_checkpoint = True
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as exc:
        print(f"Unable to create checkpoint directory '{checkpoint_path.parent}': {exc}. Skipping checkpoint save.")
        can_save_checkpoint = False

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        for images, tactile, target in train_loader:
            images, tactile, target = images.to(device), tactile.to(device), target.to(device).float()
            optimizer.zero_grad()
            pred = model(images, tactile).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        # how well the model fits training data 
        train_loss = epoch_loss / len(train_loader.dataset)
        # how well the model generalizes to unseen data 
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if wandb_module is not None:
            wandb_module.log(
                {"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss, "lr": scheduler.get_last_lr()[0]}
            )
        print(f"Epoch {epoch:03d}: train {train_loss:.4f} / val {val_loss:.4f}")

    if can_save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)
        if wandb_module is not None:
            wandb_module.save(str(checkpoint_path))
    if wandb_module is not None:
        wandb_module.finish()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, tactile, target in loader:
        images, tactile, target = images.to(device), tactile.to(device), target.to(device).float()
        pred = model(images, tactile).squeeze(-1)
        loss = criterion(pred, target)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multimodal force prediction model.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    args = parser.parse_args()

    configuration = load_config(args.config)
    train(configuration)
