# calibration-diffusion
FREE-VIEW ROBOT MANIPULATION: VISUOMOTOR POLICY BY CALIBRATION DIFFUSION


## Dataset
It is strongly recommended to install the Isaac Sim simulation environment first.

Downloads [Free-View Datasets](https://drive.google.com/file/d/1AoNEne1bn9rvioW0xzyE11pE74qRLxM2/view?usp=drive_link)


## Training
### Training Process
The training pipeline includes two sequential stages.

**Stage 1: Train Baseline Diffusion Policy**
Modify the following parameters in **train_diffusion_policy.py**:

```
task_name = "Franka_Arrange_Bottle"
dataset_path = "/media/jtl/ZJRR8/Free_View_DataSet" + task_name
```

Execute the training command:
```
python train_diffusion_policy.py
```

**Stage 2: Train Calibration Diffusion Policy**
Modify the configurations (including task_name, dataset_path and diffusion_model) in train_calibration_diffusion_policy_modify.py:
```
task_name = "Franka_Arrange_Bottle"
dataset_path = "/media/jtl/ZJRR8/Free_View_DataSet" + task_name
diffusion_model = 'xxx.pth'
```
Run the training script:
```
python train_calibration_diffusion_policy_modify.py
```
**Important Note**:
The same task must be used for both two stages.

## Test

Before running the test, you need to install the Isaac Sim environment.

The experimental environment uses Isaac Sim 4.0.0.
Different simulator versions may cause slight differences in performance and execution.

```
conda activate isaac_sim
source ~/.local/share/ov/pkg/isaac-sim-4.0.0/setup_conda_env.sh
```

Every test task is written as a separate file, stored in the Simulation_Test folder.

Some environments require loading additional USD files, which are stored in the objects/usd folder.
For example, when evaluating the **Arrange_Bottle** task, modify the following path:
```
model_name = '**.pth' ## Replace it with your trained local model path.  
usd_home = '/home/~/calibration-diffusion/objects/usd/' # Replace it with your own local path.
```


Run the test script:
```
python Simulation_Test/Arrange_Bottle/Franka_Calibration_Diffusion_Arrange_Bottle.py
```


