"""
Utilities for running the tactile gripper model during real-robot rollouts.

This module keeps the runtime dependencies minimal while matching the exact
preprocessing used in training:
  * RGB wrist image → float32 in [0,1], CHW layout, resized to 224×224.
  * Tactile history → last N timesteps (default 50), padded at the front if
    the buffer is shorter, shaped (N, 6).

Typical usage inside ``droid-multi-modal/scripts/1-pi0.py``::

    from tactile_module.robot_inference_adapter import TactileGripperAdapter

    adapter = TactileGripperAdapter(
        checkpoint_path=\"/home/pi0/multi-modal/tactile_module/checkpoints/delta_gripper.pt\",
        config_path=\"/home/pi0/multi-modal/tactile_module/configs/default.yaml\",
    )

    ...
    wrist_rgb_left = curr_obs[\"wrist_image_left\"]      # np.ndarray H×W×3, uint8 or float
    wrist_rgb_right = curr_obs[\"wrist_image_right\"]    # np.ndarray H×W×3, uint8 or float
    tactile_history = tactile_reader.read_values()      # np.ndarray (T,6) from the ring buffer
    pi0_action = pred_action_chunk[actions_from_chunk_completed]
    current_gripper = float(curr_obs[\"gripper_position\"][0])

    merged_action, tactile_delta = adapter.override_gripper(
        pi0_action,
        wrist_rgb_left,
        wrist_rgb_right,
        tactile_history,
        current_gripper,
        pi0_gate=0.2,                 # only override when pi0 meaningfully moves the gripper
        absolute_clip=(-1.0, 1.0),    # RobotEnv with gripper_action_space=\"position\" expects [-1,1]
        step_index=0,                 # use first step from action chunk
    )
    env.step(merged_action)

You can also call ``predict_delta`` or ``predict_action_chunk`` directly if you want to handle gating yourself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import yaml

from model import MultimodalForceTransformer, MultimodalTransformerConfig


@dataclass
class AdapterConfig:
    """Runtime knobs for the real-robot adapter."""

    image_size: int = 224
    tactile_length: int = 50
    tactile_channels: int = 6
    tactile_pad_value: float = 0.0


class TactileGripperAdapter:
    """Lightweight wrapper around :class:`MultimodalForceTransformer` for live control."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Args:
            checkpoint_path: Path to the trained ``delta_gripper`` weights.
            config_path: YAML config to keep image/tactile shapes in sync with training.
            device: Torch device string; defaults to CUDA when available else CPU.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config_path = Path(config_path) if config_path is not None else None

        adapter_cfg, model_cfg = self._load_configs(config_path)
        self.adapter_cfg = adapter_cfg

        model = MultimodalForceTransformer(model_cfg)
        state = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        # Remove missing keys that may not exist in older checkpoints
        model_state = model.state_dict()
        missing_keys = set(state.keys()) - set(model_state.keys())
        extra_keys = set(model_state.keys()) - set(state.keys())
        if missing_keys:
            import logging
            logging.getLogger(__name__).warning(f"Missing keys in checkpoint (will be ignored): {missing_keys}")
        if extra_keys:
            import logging
            logging.getLogger(__name__).warning(f"Extra keys in model (will use default initialization): {extra_keys}")
            # Remove extra keys from state dict to allow loading
            state_filtered = {k: v for k, v in state.items() if k in model_state}
            state = state_filtered
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model = model

    def _load_configs(self, config_path: Path | None) -> Tuple[AdapterConfig, MultimodalTransformerConfig]:
        if config_path is None or not config_path.exists():
            return AdapterConfig(), MultimodalTransformerConfig()

        with config_path.open("r") as handle:
            cfg = yaml.safe_load(handle)

        model_section = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        adapter_cfg = AdapterConfig(
            image_size=int(cfg.get("robomimic", {}).get("image_size", 224)),
            tactile_length=int(cfg.get("robomimic", {}).get("tactile_length", 50)),
            tactile_channels=int(cfg.get("robomimic", {}).get("tactile_channels", 6)),
            tactile_pad_value=float(cfg.get("robomimic", {}).get("tactile_pad_value", 0.0)),
        )

        model_cfg = MultimodalTransformerConfig(
            **{k: v for k, v in model_section.items() if v is not None},
        )
        return adapter_cfg, model_cfg

    # ----------------------------- Preprocessing helpers ----------------------------- #
    def _prepare_image(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert H×W×3 or 3×H×W image to normalised float tensor on the target device."""
        if isinstance(image, torch.Tensor):
            tensor = image.clone()
        else:
            tensor = torch.from_numpy(np.array(image))

        if tensor.dim() == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        if tensor.dim() != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected image with shape (3,H,W) or (H,W,3); got {tuple(tensor.shape)}.")

        tensor = tensor.float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0  # uint8 → [0,1]

        # Resize here so the model does not spend time interpolating every step.
        if tensor.shape[1] != self.adapter_cfg.image_size or tensor.shape[2] != self.adapter_cfg.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.adapter_cfg.image_size, self.adapter_cfg.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor.unsqueeze(0).to(self.device)  # add batch dim

    def _prepare_tactile(self, tactile_history: Iterable[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Take the most recent tactile history and pad/trim to the configured length.
        Expects an array shaped (time, channels) or flat vector length time*channels.
        """
        tactile_np = np.array(tactile_history, dtype=np.float32)
        if tactile_np.ndim == 1:
            # Flattened vector from the websocket payload.
            channels = self.adapter_cfg.tactile_channels
            if tactile_np.size % channels != 0:
                raise ValueError(
                    f"Tactile vector of length {tactile_np.size} is not divisible by channels={channels}."
                )
            tactile_np = tactile_np.reshape(-1, channels)
        elif tactile_np.ndim != 2:
            raise ValueError(f"Expected tactile history with shape (time, channels); got {tactile_np.shape}.")

        if tactile_np.shape[1] != self.adapter_cfg.tactile_channels:
            raise ValueError(
                f"Tactile channel mismatch: expected {self.adapter_cfg.tactile_channels}, got {tactile_np.shape[1]}."
            )

        if tactile_np.shape[0] < self.adapter_cfg.tactile_length:
            pad = np.full(
                (self.adapter_cfg.tactile_length - tactile_np.shape[0], self.adapter_cfg.tactile_channels),
                self.adapter_cfg.tactile_pad_value,
                dtype=np.float32,
            )
            tactile_np = np.concatenate([pad, tactile_np], axis=0)
        elif tactile_np.shape[0] > self.adapter_cfg.tactile_length:
            tactile_np = tactile_np[-self.adapter_cfg.tactile_length :]

        tactile_tensor = torch.from_numpy(tactile_np).unsqueeze(0).to(self.device)
        return tactile_tensor

    # ----------------------------- Public API ----------------------------- #
    def predict_action_chunk(
        self, 
        image_left: np.ndarray | torch.Tensor, 
        image_right: np.ndarray | torch.Tensor,
        tactile_history
    ) -> np.ndarray:
        """
        Return the action chunk (next N steps) predicted by the tactile model.
        
        Returns:
            Action chunk array of shape (action_chunk_size,) with gripper deltas for future steps.
        """
        image_left_tensor = self._prepare_image(image_left)
        image_right_tensor = self._prepare_image(image_right)
        tactile_tensor = self._prepare_tactile(tactile_history)

        with torch.no_grad():
            pred = self.model(image_left_tensor, image_right_tensor, tactile_tensor)
        return pred[0].cpu().numpy()  # Remove batch dimension
    
    def predict_delta(
        self, 
        image_left: np.ndarray | torch.Tensor, 
        image_right: np.ndarray | torch.Tensor,
        tactile_history,
        step_index: int = 0
    ) -> float:
        """
        Return a single gripper delta from the action chunk (for backward compatibility).
        
        Args:
            image_left: Left wrist camera image.
            image_right: Right wrist camera image.
            tactile_history: Latest tactile readings.
            step_index: Which step from the action chunk to return (default: 0, first step).
        
        Returns:
            Scalar gripper delta for the specified future step.
        """
        action_chunk = self.predict_action_chunk(image_left, image_right, tactile_history)
        return float(action_chunk[step_index])

    def override_gripper(
        self,
        pi0_action: np.ndarray,
        image_left: np.ndarray | torch.Tensor,
        image_right: np.ndarray | torch.Tensor,
        tactile_history,
        current_gripper_position: float,
        *,
        pi0_gate: float = 0.1,
        absolute_clip: Tuple[float, float] | None = (-1.0, 1.0),
        step_index: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Merge pi0's action with the tactile model by swapping out the gripper dimension.

        Args:
            pi0_action: The raw action vector from pi0 (shape (..., 8)).
            image_left: Left wrist camera RGB frame.
            image_right: Right wrist camera RGB frame.
            tactile_history: Latest tactile readings (time×6 or flattened).
            current_gripper_position: Scalar gripper position from robot state.
            pi0_gate: Only override when |pi0_gripper| >= gate; avoids touching idle/open commands.
            absolute_clip: Clamp the final gripper command; set to ``None`` to disable clipping.
            step_index: Which step from the action chunk to use (default: 0, first step).
        Returns:
            merged_action: Copy of ``pi0_action`` with the last dimension replaced.
            tactile_delta: Raw delta predicted by the tactile model for the selected step.
        """
        if pi0_action.shape[-1] != 8:
            raise ValueError(f"Expected pi0 action with 8 dimensions, got shape {pi0_action.shape}.")

        pi0_gripper_cmd = float(np.asarray(pi0_action)[-1])
        tactile_delta = self.predict_delta(image_left, image_right, tactile_history, step_index=step_index)
        target = current_gripper_position + tactile_delta

        if absolute_clip is not None:
            target = float(np.clip(target, absolute_clip[0], absolute_clip[1]))

        merged = np.asarray(pi0_action, dtype=np.float32).copy()
        if abs(pi0_gripper_cmd) >= pi0_gate:
            merged[-1] = target
        else:
            # Keep pi0's gripper output if it is essentially idle/open.
            merged[-1] = pi0_gripper_cmd

        return merged, float(tactile_delta)


__all__ = ["TactileGripperAdapter", "AdapterConfig"]
