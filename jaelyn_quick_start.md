## Run this command to start collecting data 
```
conda activate droid 
cd /home/pi0/multi-modal/droid-multi-modal/scripts
python jaelyn_demo.py
```

## Data Store here 
```
/home/pi0/multi-modal/droid-multi-modal/data/success
```

## Run this command to converting data into robomimic dataset 
### Convert svo to mp4 
```
conda activate droid 
cd /home/pi0/multi-modal/droid-multi-modal/scripts/convert
python remove_special_characters.py
python svo_to_mp4.py
```

### Convert to robomimic & filter out idle state 
```
cd /home/pi0/multi-modal/data-builder-multi-modal/droid
conda activate rlds_env
python robomimic_delta_gripper_v2.py
```

## Run this command to start training 
```python
./scripts/train.sh --config configs/default.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python train_tactile_gripper.py --config configs/tactile_config_dinov3.yaml
```


<!-- <!-- No need to run separately 
### Filter out idle state 
```
cd /home/pi0/multi-modal/data-builder-multi-modal/droid
python gripper_filter.py -->
``` -->