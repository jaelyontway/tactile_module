# Data Collection 
## Ensure fingers are tight, can be fastened using allen key 
### Please rerun python commands in both terminal 1 and terminal 2 after killed terminal 2 
## Terminal 1 -- Enable Gripper & Robot
```
ssh bob@172.16.0.3
conda activate polymetis-local
cd polymetis_source/R2D2
sudo chmod 666 /dev/ttyUSB0 # give access to robotiq gripper
python scripts/server/run_server.py
```
## Terminal 2 Run this command to start collecting data 
```
conda activate droid 
sudo chmod 666 /dev/ttyACM0 # give access to tactile sensing 
cd /home/pi0/multi-modal/droid-multi-modal/scripts
python jaelyn_demo.py
```
Make sure there is no error with keyword "Gripper" in Terminal 1 and then continue data collection on GUI

## Data Collection GUI
rename as your preference 
type in prompt 
Calibrate as needed, the calibration board is on the messy table 
Collect --> Proceed (tactile sensor reinitiaze everytime this button got hit) --> Press keyboard 'a' --> Press keyboard 'e' --> If save this demo, press keyboard 'a"; if discard this demo, press keyboard 's' --> Back --> Proceed --> ...


## Data Store here 
```
/home/pi0/multi-modal/droid-multi-modal/data/success/todays-date
```

# Post-Processing Data 
## Run this command to converting data into robomimic dataset 
### Convert svo to mp4 
```
conda activate droid 
cd /home/pi0/multi-modal/droid-multi-modal/scripts/convert 
python remove_special_characters.py     # don't forget hardcode the data path into this file 
python svo_to_mp4.py                    # don't forget hardcode the data path into this file 
```

### Convert to robomimic & filter out idle state 
```
cd /home/pi0/multi-modal/data-builder-multi-modal/droid
conda activate rlds_env
python robomimic_delta_gripper_v2.py
```

# Tactile Module Training 
## Run this command to start training 
```python
./scripts/train.sh --config configs/default.yaml

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python train_tactile_gripper.py --config configs/tactile_config_dinov3.yaml
```

# Tactile Module + PI0 Inference
## Run this command to start robot inference 
```
# Terminal 1 -- listen to droid policy 
cd /home/pi0/multi-modal/openpi-multi-modal/scripts
uv run serve_policy.py --env BASE 

# Terminal 2 
cd /home/pi0/multi-modal/droid-multi-modal/scripts
conda activate droid-tact
python tactile_module_inference.py
```


<!-- 
Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed -->

<!-- <!-- No need to run separately 
### Filter out idle state 
```
cd /home/pi0/multi-modal/data-builder-multi-modal/droid
python gripper_filter.py -->
``` -->
