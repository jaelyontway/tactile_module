import argparse
import logging
import math
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

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


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
    """Dataset wrapper for Robomimic-format HDF5 files."""

    LOGGED_DATASETS: set[str] = set()

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

        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency provided via requirements
            raise RuntimeError(
                "h5py is required to read Robomimic datasets. Install it via `pip install h5py`."
            ) from exc

        self.image_key = image_key
        self.tactile_key = tactile_key
        self.force_key = force_key
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.tactile_channels = int(tactile_channels)
        self.tactile_length = int(tactile_length)
        self.tactile_pad_value = float(tactile_pad_value)
        self.tactile_window = int(tactile_window) if tactile_window is not None else self.tactile_length

        self._indices: list[tuple[str, int]] = []
        with h5py.File(self.hdf5_path, 'r') as handle:
            data_group = handle['data']
            for episode_key in data_group.keys():
                trajectory = data_group[episode_key]
                force_dataset = self._resolve_dataset(trajectory, self.force_key)
                horizon = int(force_dataset.shape[0])
                for timestep in range(horizon):
                    self._indices.append((episode_key, timestep))

        if not self._indices:
            raise ValueError(f"No samples found in Robomimic dataset '{self.hdf5_path}'.")
        key_signature = f"{self.hdf5_path}:{self.image_key}:{self.tactile_key}:{self.force_key}"
        if key_signature not in self.LOGGED_DATASETS:
            episode_key, timestep = self._indices[0]
            with h5py.File(self.hdf5_path, 'r') as handle:
                trajectory = handle['data'][episode_key]
                image_shape = np.array(self._resolve_dataset(trajectory, self.image_key)[timestep]).shape
                tactile_shape = np.array(self._resolve_dataset(trajectory, self.tactile_key)).shape
                force_shape = np.array(self._resolve_dataset(trajectory, self.force_key)[timestep]).shape
            logger.info(
                "[RobomimicForceDataset] image_key='%s' shape=%s, tactile_key='%s' shape=%s, force_key='%s' shape=%s",
                self.image_key,
                image_shape,
                self.tactile_key,
                tactile_shape,
                self.force_key,
                force_shape,
            )
            self.LOGGED_DATASETS.add(key_signature)

    @staticmethod
    def _resolve_dataset(group, key_path: str):
        node = group
        for key in key_path.split('/'):
            if not key:
                continue
            node = node[key]
        return node

    def _load_image(self, trajectory, timestep: int) -> torch.Tensor:
        image_np = np.array(self._resolve_dataset(trajectory, self.image_key)[timestep])
        if image_np.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got shape {image_np.shape}.")

        if image_np.shape[0] == 3:
            image = torch.from_numpy(image_np).float()
        else:
            image = torch.from_numpy(np.transpose(image_np, (2, 0, 1))).float()

        if self.normalize_images and image.max() > 1.5:
            image = image / 255.0

        if self.image_size is not None and (
            image.shape[1] != self.image_size or image.shape[2] != self.image_size
        ):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return image.contiguous()

    def _load_tactile(self, trajectory, timestep: int) -> torch.Tensor:
        tactile_dataset = self._resolve_dataset(trajectory, self.tactile_key)
        tactile_dataset = np.array(tactile_dataset)

        if tactile_dataset.ndim == 3:
            tactile_slice = tactile_dataset[timestep]
        else:
            window = min(self.tactile_window, tactile_dataset.shape[0])
            start = max(0, timestep + 1 - window)
            tactile_slice = tactile_dataset[start : timestep + 1]

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
        force_scalar = float(force_value.reshape(-1)[-1])
        return torch.tensor(force_scalar, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - should already be installed
            raise RuntimeError(
                "h5py is required to read Robomimic datasets. Install it via `pip install h5py`."
            ) from exc

        episode_key, timestep = self._indices[index]
        with h5py.File(self.hdf5_path, 'r') as handle:
            trajectory = handle['data'][episode_key]
            image = self._load_image(trajectory, timestep)
            tactile = self._load_tactile(trajectory, timestep)
            force = self._load_force(trajectory, timestep)
        return image, tactile, force

    def __len__(self) -> int:
        return len(self._indices)


class RegressionMetricAccumulator:
    """Running accumulator for scalar regression metrics."""

    def __init__(self) -> None:
        self.count = 0
        self.abs_error = 0.0
        self.sq_error = 0.0
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred_sq = 0.0
        self.sum_target_sq = 0.0
        self.sum_prod = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        pred_flat = prediction.detach().reshape(-1)
        target_flat = target.detach().reshape(-1)
        diff = pred_flat - target_flat

        self.count += target_flat.numel()
        self.abs_error += diff.abs().sum().item()
        self.sq_error += diff.pow(2).sum().item()
        self.sum_pred += pred_flat.sum().item()
        self.sum_target += target_flat.sum().item()
        self.sum_pred_sq += pred_flat.pow(2).sum().item()
        self.sum_target_sq += target_flat.pow(2).sum().item()
        self.sum_prod += (pred_flat * target_flat).sum().item()

    def results(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "count": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "pearson": 0.0,
            }

        mse = self.sq_error / self.count
        mae = self.abs_error / self.count
        rmse = math.sqrt(mse)

        pred_mean = self.sum_pred / self.count
        target_mean = self.sum_target / self.count
        pred_var = max(self.sum_pred_sq / self.count - pred_mean ** 2, 0.0)
        target_var = max(self.sum_target_sq / self.count - target_mean ** 2, 0.0)
        ss_tot = self.sum_target_sq - self.count * (target_mean ** 2)
        r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - self.sq_error / ss_tot

        covariance = self.sum_prod / self.count - pred_mean * target_mean
        denom = math.sqrt(max(pred_var * target_var, 0.0))
        pearson = covariance / denom if denom > 1e-12 else 0.0
        pearson = max(min(pearson, 1.0), -1.0)

        return {
            "count": float(self.count),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson,
        }


def _grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Compute gradient norm without modifying gradients."""
    valid_params = [p for p in parameters if p.grad is not None]
    if not valid_params:
        return 0.0

    if math.isinf(norm_type):
        return max(p.grad.detach().abs().max().item() for p in valid_params)

    total = 0.0
    for param in valid_params:
        param_norm = param.grad.detach().data.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type)


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

    float_keys = {"lr", "weight_decay", "dropout", "grad_clip_norm"}
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

    grad_clip_norm = None
    grad_clip_value = config.get("grad_clip_norm")
    if grad_clip_value not in (None, "", 0):
        try:
            grad_clip_norm = float(grad_clip_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"grad_clip_norm must be numeric, got {grad_clip_value!r}") from exc
        if grad_clip_norm <= 0:
            grad_clip_norm = None

    grad_norm_type_value = config.get("grad_norm_type", 2.0)
    try:
        grad_norm_type = float(grad_norm_type_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grad_norm_type must be numeric, got {grad_norm_type_value!r}") from exc

    dataset_type = config.get("dataset_type", "dummy").lower()
    logged_messages: set[str] = set()

    def log_once(identifier: str, message: str) -> None:
        if identifier not in logged_messages:
            print(message)
            logged_messages.add(identifier)

    def build_dataset(split: str):
        if dataset_type == "robomimic":
            dataset_cfg = config.get("robomimic", {})
            path_key = "train_path" if split == "train" else "val_path"
            hdf5_path = dataset_cfg.get(path_key)
            if not hdf5_path:
                raise ValueError(f"Missing robomimic.{path_key} path in configuration.")

            image_key = dataset_cfg.get("image_key")
            tactile_key = dataset_cfg.get("tactile_key")
            force_key = dataset_cfg.get("force_key")
            if not image_key or not tactile_key or not force_key:
                raise ValueError(
                    "Robomimic configuration must provide 'image_key', 'tactile_key', and 'force_key'."
                )

            image_size_value = dataset_cfg.get("image_size", config.get("dummy_image_size"))
            try:
                image_size = int(image_size_value) if image_size_value not in (None, "", 0) else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.image_size value: {image_size_value!r}") from exc

            tactile_length_value = dataset_cfg.get("tactile_length", config.get("dummy_tactile_length"))
            try:
                tactile_length = int(tactile_length_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_length value: {tactile_length_value!r}") from exc

            tactile_window_value = dataset_cfg.get("tactile_window")
            if tactile_window_value in (None, "", 0):
                tactile_window = None
            else:
                try:
                    tactile_window = int(tactile_window_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid robomimic.tactile_window value: {tactile_window_value!r}") from exc

            tactile_pad_value_value = dataset_cfg.get("tactile_pad_value", 0.0)
            try:
                tactile_pad_value = float(tactile_pad_value_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_pad_value value: {tactile_pad_value_value!r}") from exc

            tactile_channels_value = dataset_cfg.get("tactile_channels", 6)
            try:
                tactile_channels = int(tactile_channels_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid robomimic.tactile_channels value: {tactile_channels_value!r}") from exc
            normalize_images = bool(dataset_cfg.get("normalize_images", True))

            dataset = RobomimicForceDataset(
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
            log_once(
                f"robomimic-{split}",
                (
                    f"[train] Using Robomimic dataset ({split}) from {dataset.hdf5_path} "
                    f"(image='{dataset.image_key}', tactile='{dataset.tactile_key}', force='{dataset.force_key}')"
                ),
            )
            return dataset

        use_dummy = (
            config.get("use_dummy_data", False)
            or ForceDataset is None
            or dataset_type == "dummy"
        )
        if not use_dummy:
            if ForceDataset is None:
                raise RuntimeError("ForceDataset is unavailable. Enable 'use_dummy_data' or provide an implementation.")
            dataset = ForceDataset(split=split)
            log_once(f"external-{split}", "[train] Using user-provided ForceDataset implementation.")
            return dataset

        length = config["dummy_train_size"] if split == "train" else config["dummy_val_size"]
        dataset = DummyForceDataset(
            split=split,
            length=length,
            image_size=config["dummy_image_size"],
            tactile_len=config["dummy_tactile_length"],
            seed=config.get("dummy_seed", 0),
        )
        log_once(
            f"dummy-{split}",
            (
                f"[train] Using dummy dataset ({split}) with length={length}, "
                f"image_size={config['dummy_image_size']}, tactile_len={config['dummy_tactile_length']}."
            ),
        )
        return dataset

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
        train_metrics = RegressionMetricAccumulator()
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_norm_count = 0
        grad_clip_events = 0

        for images, tactile, target in train_loader:
            images, tactile, target = images.to(device), tactile.to(device), target.to(device).float()
            optimizer.zero_grad()
            pred = model(images, tactile).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()

            grad_norm_count += 1
            if grad_clip_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                grad_norm_value = float(total_norm)
                if grad_norm_value > grad_clip_norm:
                    grad_clip_events += 1
            else:
                grad_norm_value = _grad_norm(model.parameters(), norm_type=grad_norm_type)

            grad_norm_sum += grad_norm_value
            grad_norm_max = max(grad_norm_max, grad_norm_value)

            optimizer.step()
            train_metrics.update(pred.detach(), target.detach())

        train_results = train_metrics.results()
        val_results = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        grad_norm_mean = grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
        grad_clip_rate = grad_clip_events / grad_norm_count if grad_norm_count else 0.0

        if wandb_module is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": train_results["mse"],
                "train/mae": train_results["mae"],
                "train/rmse": train_results["rmse"],
                "train/r2": train_results["r2"],
                "train/pearson": train_results["pearson"],
                "val/loss": val_results["mse"],
                "val/mae": val_results["mae"],
                "val/rmse": val_results["rmse"],
                "val/r2": val_results["r2"],
                "val/pearson": val_results["pearson"],
                "lr": scheduler.get_last_lr()[0],
                "train/grad_norm_mean": grad_norm_mean,
                "train/grad_norm_max": grad_norm_max,
                "train/grad_clip_rate": grad_clip_rate,
            }
            if grad_clip_norm is not None:
                log_payload["train/grad_clip_threshold"] = grad_clip_norm
            wandb_module.log(log_payload)

        clip_msg = ""
        if grad_clip_norm is not None and grad_norm_count:
            clip_msg = f" | clip {grad_clip_rate * 100:.1f}%"
        print(
            (
                f"Epoch {epoch:03d}: "
                f"train mse {train_results['mse']:.4f} mae {train_results['mae']:.4f} "
                f"| val mse {val_results['mse']:.4f} mae {val_results['mae']:.4f} "
                f"| grad {grad_norm_mean:.2f}{clip_msg}"
            )
        )

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
    metrics = RegressionMetricAccumulator()
    for images, tactile, target in loader:
        images, tactile, target = images.to(device), tactile.to(device), target.to(device).float()
        pred = model(images, tactile).squeeze(-1)
        _ = criterion(pred, target)  # keep parity with training for potential custom losses
        metrics.update(pred.detach(), target.detach())
    return metrics.results()


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
