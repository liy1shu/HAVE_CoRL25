# Learn from What We HAVE: History-Aware VErifier that Reasons about Past Interactions Online

pip install -e .

## Overall Pipeline

1. Generator: Should be able to train PNDiT
2. Dataset: Should be able to generate dataset for training HAVE
3. Verifier: Should be able to train HAVE
4. Evaluate: Run simulation

## Train the Generator

python scripts/train_generator.py

Modify the dataset item in configs/train.yaml to choose different dataset.

### Set up DELTA tracking

submodule in src/have/utils/DELTA
put the checkpoint in src/have/utils/DELTA/checkpoints


## Generate dataset to train verifier

python scripts/collect_dataset.py --output_path /data/failure_dataset_door/ --num_processes 4 --traj_per_joint_train 1 --traj_per_joint_test 1 --max_trials_train 1 --max_trials_test 1

python scripts/collect_dataset.py --dataset_path /home/yishu/datasets/failure_history_door/raw/ --output_path /data/failure_dataset_door/ --num_processes 4 --traj_per_joint_train 1 --traj_per_joint_test 1 --max_trials_train 1 --max_trials_test 1

--door is when you only want to train on multimodal door.

Then you need to run preprocess_dataset.py to split the dataset into different splits - either train / held-out / doors


python scripts/preprocess_dataset.py --split_name train_val --dataset_path /data/failure_dataset_door/ --num_processes 4

python scripts/preprocess_dataset.py --split_name held_out --dataset_path /data/failure_dataset_door/ --num_processes 4

- Uneven object

python scripts/collect_dataset.py --output_path /data/failure_dataset_uneven_release_check/ --num_processes 4 --traj_per_joint_train 1 --max_trials_train 1 --uneven

## Train verifier

CUDA_VISIBLE_DEVICES=0,1 accelerate launch scripts/train_verifier.py

