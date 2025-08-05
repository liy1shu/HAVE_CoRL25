import os
import json
import pickle as pkl
import tqdm
import random
import argparse

parser = argparse.ArgumentParser(description='Generate URDF files with for Multimodal Doors.')

# Generation parameters
parser.add_argument('--num_processes', type=int, default=16, help='Number of processes used for data generation.')
parser.add_argument('--dataset_path', type=str, default='/data/failure_dataset_door/', help='The path of the dataset.')
parser.add_argument('--split_name', type=str, default='train_val', help='Which split to use (train_val, held_out, multimodal_door).')
args = parser.parse_args()


dataset_path = args.dataset_path
process_num = args.num_processes
split_name = args.split_name # train_val, held-out, multimodal_door
assert split_name in ['train_val', 'held_out', 'multimodal_door'], "Only support train_val, held_out, multimodal_door splits!"

train_data_id = 0
train_curr_sample_idx = 0
train_sample_idx_to_data_idx = {}
train_sample_idx_to_action_idx = {}

val_data_id = 0
val_curr_sample_idx = 0
val_sample_idx_to_data_idx = {}
val_sample_idx_to_action_idx = {}

for i in tqdm.tqdm(range(process_num)):
    if not os.path.exists(os.path.join(dataset_path, split_name, 'train')):
        os.makedirs(os.path.join(dataset_path, split_name, 'train'), exist_ok=True)
    if not os.path.exists(os.path.join(dataset_path, split_name, 'val')):
        os.makedirs(os.path.join(dataset_path, split_name, 'val'), exist_ok=True)

    if split_name == 'train_val' or split_name == 'multimodal_door':  # Train val split (no held-out categories, just randomly split the trajectories of the train object instances)
        with open(os.path.join(dataset_path, f'raw_pkls/train_{i}.pkl'), 'rb') as f:
            trajectories = pkl.load(f)
    
        random.shuffle(trajectories)

        split_idx = int(4 / 5 * len(trajectories)) 
        train_data = trajectories[:split_idx] 
        val_data = trajectories[split_idx:]

        for trajectory in tqdm.tqdm(train_data):
            # Save the train trajectory
            for action_idx in range(len(trajectory["evaluate"]["pcds"])):
                train_sample_idx_to_data_idx[train_curr_sample_idx] = train_data_id
                train_sample_idx_to_action_idx[train_curr_sample_idx] = action_idx
                train_curr_sample_idx += 1
                # break  # Just keep one for sampler

            with open(os.path.join(dataset_path, f'{split_name}/train/{train_data_id}.pkl'), 'wb') as f:
                pkl.dump(trajectory, f)
            
            train_data_id += 1

        for trajectory in tqdm.tqdm(val_data):
            for action_idx in range(len(trajectory["evaluate"]["pcds"])):
                val_sample_idx_to_data_idx[val_curr_sample_idx] = val_data_id
                val_sample_idx_to_action_idx[val_curr_sample_idx] = action_idx
                val_curr_sample_idx += 1
                # break  # Just keep one for sampler

            with open(os.path.join(dataset_path, f'{split_name}/val/{val_data_id}.pkl'), 'wb') as f:
                pkl.dump(trajectory, f)
                
            val_data_id += 1
    
    
    elif split_name == 'held_out':  # Train val unseen split (held-out categories, split the trajectories of the train object categories into train and val sets)
        with open(f'{dataset_path}/raw_pkls/train_{i}.pkl', 'rb') as f:
            trajectories = pkl.load(f)
        with open(f'{dataset_path}/raw_pkls/test_{i}.pkl', 'rb') as f:
            trajectories += pkl.load(f)
        
        # Split based on the held-out category experiment splits
        with open('./metadata/movable_links_fullset_000_full.json', 'r') as f:
            movable_links = json.load(f)

        with open('./metadata/articulated_heldout.json', 'r') as f:
            data_split = json.load(f)


        for trajectory in tqdm.tqdm(trajectories):
            # In train set
            if trajectory['obj_id'] in data_split['train-train']:
                # Save the train trajectory
                for action_idx in range(len(trajectory["evaluate"]["pcds"])):
                    train_sample_idx_to_data_idx[train_curr_sample_idx] = train_data_id
                    train_sample_idx_to_action_idx[train_curr_sample_idx] = action_idx
                    train_curr_sample_idx += 1
                    # break  # Just keep one for sampler

                # valid_id += 1  # If its dataset, there is no validness check, just use data_id
                with open(os.path.join(dataset_path, f'{split_name}/train/{train_data_id}.pkl'), 'wb') as f:
                    pkl.dump(trajectory, f)
                
                train_data_id += 1
            
            elif trajectory['obj_id'] in data_split['train-test']:  # In val set
                for action_idx in range(len(trajectory["evaluate"]["pcds"])):
                    val_sample_idx_to_data_idx[val_curr_sample_idx] = val_data_id
                    val_sample_idx_to_action_idx[val_curr_sample_idx] = action_idx
                    val_curr_sample_idx += 1
                    # break  # Just keep one for sampler

                # valid_id += 1   # If its dataset, there is no validness check, just use data_id
                with open(os.path.join(dataset_path, f'{split_name}/val/{val_data_id}.pkl'), 'wb') as f:
                    pkl.dump(trajectory, f)
                
                val_data_id += 1


    else:
        print("Error: unknown split name!")



train_sample_idx_to_everything = {
    "data_idx": train_sample_idx_to_data_idx,
    "action_idx": train_sample_idx_to_action_idx
}

import json
with open(os.path.join(dataset_path, split_name, 'train/indices.json'), 'w') as f:
    json.dump(train_sample_idx_to_everything, f)


val_sample_idx_to_everything = {
    "data_idx": val_sample_idx_to_data_idx,
    "action_idx": val_sample_idx_to_action_idx
}

import json
with open(os.path.join(dataset_path, split_name, 'val/indices.json'), 'w') as f:
    json.dump(val_sample_idx_to_everything, f)
