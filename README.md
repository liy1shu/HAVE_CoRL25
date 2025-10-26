# Learn from What We HAVE: History-Aware VErifier that Reasons about Past Interactions Online [CoRL 2025]

This is the official code repo for CoRL 2025 Paper Learn from What We HAVE.

## Conda Environment

conda create -n have python=3.9
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flowbot3d@git+https://github.com/r-pad/flowbot3d.git --no-dependencies
pip install -e .

## Overall Pipeline

1. Train the Generator
2. Generate Dataset for Verifier Training
3. Train the Verifier
4. Run Simulation Evaluation

## Train the Generator

python scripts/train_generator.py

Modify the dataset item in configs/train.yaml to choose different datasets.

### Set up DELTA tracking

Clone DELTA (git@github.com:snap-research/DELTA_densetrack3d.git) as a submodule in src/have/utils/DELTA

Download and put the DELTA checkpoint in src/have/utils/DELTA/checkpoints.


## Generate dataset to train verifier

```
python scripts/collect_dataset.py --output_path /data/failure_dataset_door/ --num_processes 4 --traj_per_joint_train 1 --traj_per_joint_test 1 --max_trials_train 1 --max_trials_test 1

python scripts/collect_dataset.py --dataset_path /home/yishu/datasets/failure_history_door/raw/ --output_path /data/failure_dataset_door/ --num_processes 4 --traj_per_joint_train 1 --traj_per_joint_test 1 --max_trials_train 1 --max_trials_test 1
```

--door is when you only want to train on multimodal door.

Then you need to run preprocess_dataset.py to split the dataset into different splits - either train / held-out / doors

```
python scripts/preprocess_dataset.py --split_name train_val --dataset_path /data/failure_dataset_door/ --num_processes 4

python scripts/preprocess_dataset.py --split_name held_out --dataset_path /data/failure_dataset_door/ --num_processes 4
```

- For Uneven object

```
python scripts/collect_dataset.py --dataset_path /data/datasets/unevenobject/raw/ --output_path /data/failure_dataset_uneven_release_check/ --num_processes 4 --traj_per_joint_train 1 --max_trials_train 1 --uneven
```

## Train verifier

```
CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/train_verifier.py
```

## Run Simulation

```
python scripts/sim_eval.py
```