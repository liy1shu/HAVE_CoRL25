import json
import os
import torch
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
import torch_geometric.data as tgd


class ActionDataset(Dataset):
    def __init__(self, subset, root_dir='/project_data/held/yishul/datasets/failure_dataset/articulated_object_random/train_val_unseen/'):
        self.directory = os.path.join(root_dir, subset)
        indices_path = os.path.join(self.directory, 'indices.json')
        count_path = os.path.join(root_dir, "obj_id_to_weight.json")
        with open(indices_path, 'r') as f:
            indices = json.load(f)

        with open(count_path, 'r') as f:
            obj_id_to_weight = json.load(f)

        # Construct a (history, action_to_evaluate) pair
        self.sample_idx_to_data_idx = indices["data_idx"]
        self.sample_idx_to_action_idx = indices["action_idx"]
        self.obj_id_to_weight = obj_id_to_weight


    def __len__(self):
        return len(self.sample_idx_to_data_idx)

    def __getitem__(self, idx):
        # import time
        idx = f"{idx}"
        # start_t = time.time()
        data_idx = self.sample_idx_to_data_idx[idx]
        with open(os.path.join(self.directory, f'{data_idx}.pkl'), 'rb') as f:
            sample = pkl.load(f)
        action_idx = self.sample_idx_to_action_idx[idx]
        # print(data_idx, len(batch_data))

        # print(action_idx, len(sample["evaluate"]["pcds"]))

        action_to_evaluate_pcd = sample["evaluate"]["pcds"][action_idx]
        action_to_evaluate_flow = sample["evaluate"]["flows"][action_idx]

        # gt_scores = 1 if sample["scores"][action_idx] > 0.6 else 0
        max_gt_score = np.max(sample["evaluate"]["scores"])
        if max_gt_score < 0.1:
            max_gt_score = 0.1
        # print(len(sample["evaluate"]["scores"]), action_idx)
        gt_scores = np.clip(sample["evaluate"]["scores"][action_idx] / max_gt_score, 0, 1) * 2 - 1  # Clipping the negative

        action_pcds = sample["pcds"][:-1]
        action_flows = sample["flows"][:-1]
        action_results = sample["obs_flows"][:-1]
        action_scores = np.clip(sample["scores"][:-1], 0, 1) * 2 - 1   # bad history: -1, good history: 1
        weight = self.obj_id_to_weight[sample["obj_id"]]
        # print(f"got data in {time.time() - start_t}!")
        return action_to_evaluate_pcd, action_to_evaluate_flow, action_pcds, action_flows, action_results, gt_scores, action_scores, weight


def collate_fn(batch):
    # List of tuples where each tuple is (history, prediction)
    action_to_evaluate_pcd, action_to_evaluate_flow, action_pcds, action_flows, action_results, gt_scores, history_scores, weights = zip(*batch)

    # Find the maximum length of history in the batch
    max_len = max(len(action_flow) for action_flow in action_flows)
                                        
    # Pad histories and create padding masks
    padded_action_pcds = torch.zeros((len(action_results), max_len, 1200, 3), dtype=torch.float)
    padded_action_flows = torch.zeros((len(action_results), max_len, 1200, 3), dtype=torch.float)
    padded_action_results = torch.zeros((len(action_results), max_len, 1200, 3), dtype=torch.float)
    padding_masks = torch.ones((len(action_results), max_len + 1), dtype=torch.bool)
    
    cnt = 0
    for i, (pcd, flow, result) in enumerate(zip(action_pcds, action_flows, action_results)):
        cnt += len(pcd)
        padded_action_pcds[i, :len(pcd)] = torch.from_numpy(pcd).float()
        padded_action_flows[i, :len(flow)] = torch.from_numpy(flow).float()
        padded_action_results[i, :len(result)] = torch.from_numpy(result).float()
        padding_masks[i, :(len(pcd) + 1)] = 0  # 0 for actual data, 1 for padding
    
    gt_scores = torch.from_numpy(np.stack(gt_scores))
    history_scores = torch.from_numpy(np.concatenate(history_scores))
    # assert history_scores.shape[0] == cnt, breakpoint()
    data_list = []
    for pcd, flow in zip(action_to_evaluate_pcd, action_to_evaluate_flow):
        data_list.append(
            tgd.Data(
                pos=torch.from_numpy(pcd).float(),
                x=torch.from_numpy(flow).float()
            )
        )
    action_to_evaluate = tgd.Batch.from_data_list(data_list)
    weights = torch.tensor(weights).float()
    # breakpoint()

    return action_to_evaluate, padded_action_pcds, padded_action_flows, padded_action_results, gt_scores, history_scores, padding_masks, weights
