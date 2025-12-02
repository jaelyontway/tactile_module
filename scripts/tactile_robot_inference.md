# Tactile-Only Robot Inference (pi0/DROID)

Use this as a drop-in guide to run the trained tactile gripper model (`checkpoints/delta_gripper.pt`) on the real robot without touching the vision-language policy. Nothing in this repo is overwritten—follow the steps and copy the snippet into your DROID script when ready.

## 1) Quick sanity check on the robot laptop
Make sure the weights load and a forward pass works on your hardware (CPU or GPU):
```bash
cd /home/pi0/multi-modal/tactile_module
python - <<'PY'
import torch, numpy as np
from tactile_module.robot_inference_adapter import TactileGripperAdapter

adapter = TactileGripperAdapter(
    checkpoint_path="/home/pi0/multi-modal/tactile_module/checkpoints/delta_gripper.pt",
    config_path="/home/pi0/multi-modal/tactile_module/configs/default.yaml",
)

# Fake inputs: 224×224 wrist image + 50×6 tactile window
image = (np.random.rand(224, 224, 3) * 255).astype("uint8")
tactile = np.random.randn(50, 6).astype("float32")

delta = adapter.predict_delta(image, tactile)
print(f"✓ Model loaded; dummy gripper delta = {delta:.4f}")
PY
```
If this runs, your checkpoint and dependencies are good to go.

## 2) Hook into the DROID control loop (no file edits done here)
Add the following snippet to your rollout script (e.g., `droid-multi-modal/scripts/1-pi0.py`) where you build the observation and send an action. The key pieces are:

```python
# Import once near the top:
import sys
sys.path.append("/home/pi0/multi-modal/tactile_module")
from tactile_module.robot_inference_adapter import TactileGripperAdapter

# After creating tactile_reader / env:
tactile_adapter = TactileGripperAdapter(
    checkpoint_path="/home/pi0/multi-modal/tactile_module/checkpoints/delta_gripper.pt",
    config_path="/home/pi0/multi-modal/tactile_module/configs/default.yaml",
)

# Inside the control loop, right after you have curr_obs and action:
wrist_rgb = curr_obs["wrist_image"]                  # H×W×3 numpy array
tactile_history = np.array(tactile_reader.read_values())  # (T,6) or flat
current_gripper = float(curr_obs["gripper_position"][0])

merged_action, tactile_delta = tactile_adapter.override_gripper(
    pi0_action=action,                   # shape (..., 8)
    image=wrist_rgb,
    tactile_history=tactile_history,
    current_gripper_position=current_gripper,
    pi0_gate=0.2,                        # only override when pi0 moves gripper
    absolute_clip=(-1.0, 1.0),           # keep RobotEnv happy
)
env.step(merged_action)
```
- `pi0_gate`: avoids fighting pi0 when it is idle/opening; lower it if you want tactile to take over more often.
- `absolute_clip`: keep the gripper command in the environment’s expected range.
- The adapter automatically pads/trims tactile to the last 50×6 and resizes the wrist RGB to 224×224 before predicting.

## 3) Run as usual
1. Start the policy server if you still use vision-language actions for the arm:  
   `python /home/pi0/multi-modal/openpi-multi-modal/scripts/serve_policy.py`
2. Launch your rollout script (after adding the snippet):  
   `python /home/pi0/multi-modal/droid-multi-modal/scripts/1-pi0.py`
3. Enter an instruction; the adapter will replace only the gripper dimension while leaving the other 7 joint commands untouched.

## 4) Troubleshooting tips
- If you see a shape error, print `tactile_history.shape`—it must be (time, 6) or a flat multiple of 6.
- If frames look washed out, ensure `curr_obs["wrist_image"]` is uint8 or floats in [0,1]; the adapter normalizes automatically.
- To log what the tactile model is doing, temporarily add: `print(f"tactile delta: {tactile_delta:.3f}")` inside the loop.
