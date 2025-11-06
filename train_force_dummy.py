import argparse
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


class RobomimicForceDataset(Dataset):
    """
    Dataset wrapper for Robomimic-format HDF5 files.

    The loader expects a file with a ``data`` group containing trajectory entries. Each
    trajectory is indexed as ``data/<episode_id>`` and exposes the same observation keys
    available during Robomimic training. The configuration selects which keys correspond
    to RGB frames, tactile sensor readings, and the force supervision signal.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        image_key: str,
        tactile_key: str,
        force_key: str,
        tactile_length: int,
        tactile_channels: int,
        tactile_pad_value: float = 0.0,
        tactile_window: Optional[int] = None,
        image_size: Optional[int] = None,
        normalize_images: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path).expanduser()
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Robomimic dataset '{self.hdf5_path}' does not exist.")

        self.image_key = image_key
        self.tactile_key = tactile_key
        self.force_key = force_key
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.tactile_length = int(tactile_length)
        self.tactile_channels = int(tactile_channels)
        self.tactile_pad_value = float(tactile_pad_value)
        self.tactile_window = int(tactile_window) if tactile_window is not None else self.tactile_length

        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency provided via requirements
            raise RuntimeError(
                "h5py is required to read Robomimic datasets. Install it via `pip install h5py`."
            ) from exc

        self._indices: list[Tuple[str, int]] = []
        with h5py.File(self.hdf5_path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"File '{self.hdf5_path}' does not contain a 'data' group.")
            data_group = handle["data"]
            for episode_key in data_group.keys():
                trajectory = data_group[episode_key]
                force_dataset = self._resolve_dataset(trajectory, self.force_key)
                horizon = int(force_dataset.shape[0])
                for timestep in range(horizon):
                    self._indices.append((episode_key, timestep))

        if not self._indices:
            raise ValueError(f"No samples found in Robomimic dataset '{self.hdf5_path}'.")

    @staticmethod
    def _resolve_dataset(group, key_path: str):
        node = group
        for key in key_path.split("/"):
            if not key:
                continue
            node = node[key]
        return node

    def __len__(self) -> int:
        return len(self._indices)

    def _load_image(self, trajectory, timestep: int) -> torch.Tensor:
        image_np = np.array(self._resolve_dataset(trajectory, self.image_key)[timestep])
        if image_np.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got shape {image_np.shape}.")

        if image_np.shape[0] == 3:
            image = torch.from_numpy(image_np).float()
        else:
            image = torch.from_numpy(np.transpose(image_np, (2, 0, 1))).float()

        if self.normalize_images:
            if image.max() > 1.5:  # assume uint8 range
                image = image / 255.0

        if self.image_size is not None and (
            image.shape[1] != self.image_size or image.shape[2] != self.image_size
        ):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return image.contiguous()

    def _load_tactile(self, trajectory, timestep: int) -> torch.Tensor:
        tactile_dataset = self._resolve_dataset(trajectory, self.tactile_key)
        try:
            dataset_shape = tactile_dataset.shape  # type: ignore[attr-defined]
        except AttributeError:
            tactile_dataset = np.array(tactile_dataset)
            dataset_shape = tactile_dataset.shape

        if len(dataset_shape) == 3:
            tactile_slice = np.array(tactile_dataset[timestep])
        else:
            window = min(self.tactile_window, dataset_shape[0])
            start = max(0, timestep + 1 - window)
            tactile_slice = np.array(tactile_dataset[start : timestep + 1])

        if tactile_slice.ndim == 1:
            tactile_slice = tactile_slice[:, None]

        if tactile_slice.ndim != 2:
            raise ValueError(
                f"Expected tactile slice with shape (time, channels); received shape {tactile_slice.shape}."
            )

        tactile_tensor = torch.from_numpy(tactile_slice).float()
        if tactile_tensor.size(-1) != self.tactile_channels:
            if tactile_tensor.size(0) == self.tactile_channels:
                tactile_tensor = tactile_tensor.transpose(0, 1)
            else:
                raise ValueError(
                    f"Tactile channels mismatch: expected {self.tactile_channels}, got {tactile_tensor.size(-1)}."
                )

        if tactile_tensor.size(0) < self.tactile_length:
            pad = torch.full(
                (self.tactile_length - tactile_tensor.size(0), self.tactile_channels),
                self.tactile_pad_value,
                dtype=torch.float32,
            )
            tactile_tensor = torch.cat([pad, tactile_tensor], dim=0)
        elif tactile_tensor.size(0) > self.tactile_length:
            tactile_tensor = tactile_tensor[-self.tactile_length :]

        return tactile_tensor.contiguous()

    def _load_force(self, trajectory, timestep: int) -> torch.Tensor:
        force_value = np.array(self._resolve_dataset(trajectory, self.force_key)[timestep])
        return torch.tensor(float(np.asarray(force_value).squeeze()), dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - should already be installed
            raise RuntimeError(
                "h5py is required to read Robomimic datasets. Install it via `pip install h5py`."
            ) from exc

        episode_key, timestep = self._indices[index]
        with h5py.File(self.hdf5_path, "r") as handle:
            trajectory = handle["data"][episode_key]
            image = self._load_image(trajectory, timestep)
            tactile = self._load_tactile(trajectory, timestep)
            force = self._load_force(trajectory, timestep)
        return image, tactile, force
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

    dataset_type = config.get("dataset_type", "dummy").lower()

    def build_dataset(split: str):
        if dataset_type == "robomimic":
            dataset_cfg = config.get("robomimic", {})
            path_key = "train_path" if split == "train" else "val_path"
            hdf5_path = dataset_cfg.get(path_key)
            if not hdf5_path:
                raise ValueError(f"Missing robomimic.{path_key} path in configuration.")

            image_key = dataset_cfg.get("image_key", "observations/images/agentview_image")
            tactile_key = dataset_cfg.get("tactile_key", "observations/tactile")
            force_key = dataset_cfg.get("force_key", "observations/force")

            image_size_value = dataset_cfg.get("image_size", config.get("dummy_image_size"))
            try:
                image_size = int(image_size_value) if image_size_value not in (None, "", 0) else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.image_size value: {image_size_value!r}") from exc

            try:
                tactile_length = int(dataset_cfg.get("tactile_length", config.get("dummy_tactile_length", 500)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_length value: {dataset_cfg.get('tactile_length')!r}") from exc

            tactile_window = dataset_cfg.get("tactile_window")
            if tactile_window is not None:
                try:
                    tactile_window = int(tactile_window)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid robomimic.tactile_window value: {tactile_window!r}") from exc
            try:
                tactile_pad_value = float(dataset_cfg.get("tactile_pad_value", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_pad_value value: {dataset_cfg.get('tactile_pad_value')!r}") from exc
            try:
                tactile_channels = int(dataset_cfg.get("tactile_channels", 6))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_channels value: {dataset_cfg.get('tactile_channels')!r}") from exc
            normalize_images = bool(dataset_cfg.get("normalize_images", True))

            return RobomimicForceDataset(
                hdf5_path=hdf5_path,
                image_key=image_key,
                tactile_key=tactile_key,
                force_key=force_key,
                tactile_length=tactile_length,
                tactile_channels=tactile_channels,
                tactile_pad_value=tactile_pad_value,
                tactile_window=tactile_window,
                image_size=image_size,
                normalize_images=normalize_images,
            )

        use_dummy = (
            config.get("use_dummy_data", False)
            or ForceDataset is None
            or dataset_type == "dummy"
        )
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
        try:
            torch.save(model.state_dict(), checkpoint_path)
            if wandb_module is not None:
                wandb_module.save(str(checkpoint_path))
        except (OSError, RuntimeError) as exc:
            print(f"Failed to save checkpoint to '{checkpoint_path}': {exc}")
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
