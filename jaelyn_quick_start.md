## Run this command to start training 

```python
./scripts/train.sh --config configs/default.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python train_force_dummy.py --config configs/tactile_config_dinov3.yaml
```
