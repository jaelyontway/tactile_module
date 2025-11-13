from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


def compute_gripper_deltas(
    gripper_sequence: np.ndarray,
    epsilon: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute timestep deltas and an active mask from a gripper position sequence.

    Args:
        gripper_sequence: Array shaped (time, ...) containing gripper positions.
        epsilon: Absolute-change threshold used to mark idle transitions.

    Returns:
        deltas: 1D array of length (time - 1) containing scalar deltas per step.
        active_mask: Boolean array marking indices where |delta| > epsilon.
    """
    if gripper_sequence.ndim < 1 or gripper_sequence.shape[0] < 2:
        logging.getLogger(__name__).warning(
            "compute_gripper_deltas received insufficient frames (shape=%s).",
            gripper_sequence.shape,
        )
        return np.array([], dtype=np.float32), np.array([], dtype=bool)

    flattened = gripper_sequence.reshape(gripper_sequence.shape[0], -1)
    deltas = flattened[1:] - flattened[:-1]
    # Use the last element (gripper command usually stored as scalar or final column).
    delta_scalar = deltas.reshape(deltas.shape[0], -1)[:, -1]
    active_mask = np.abs(delta_scalar) > float(epsilon)
    return delta_scalar.astype(np.float32), active_mask


def filter_precomputed_deltas(
    delta_sequence: np.ndarray,
    epsilon: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an active mask from pre-computed gripper deltas (e.g., from robomimic_delta_gripper.py).

    This function is used when your HDF5 file already contains delta_gripper_position.
    Unlike compute_gripper_deltas, this does NOT compute deltas—it just filters them.

    Args:
        delta_sequence: Array shaped (time, ...) containing pre-computed gripper deltas.
        epsilon: Absolute-change threshold used to mark idle transitions.

    Returns:
        deltas: 1D array of length (time) containing scalar deltas per step.
        active_mask: Boolean array marking indices where |delta| > epsilon.
    """
    if delta_sequence.ndim < 1 or delta_sequence.shape[0] < 1:
        logging.getLogger(__name__).warning(
            "filter_precomputed_deltas received empty sequence (shape=%s).",
            delta_sequence.shape,
        )
        return np.array([], dtype=np.float32), np.array([], dtype=bool)

    flattened = delta_sequence.reshape(delta_sequence.shape[0], -1)
    # Use the last element (gripper command usually stored as scalar or final column).
    delta_scalar = flattened.reshape(flattened.shape[0], -1)[:, -1]
    active_mask = np.abs(delta_scalar) > float(epsilon)
    return delta_scalar.astype(np.float32), active_mask
