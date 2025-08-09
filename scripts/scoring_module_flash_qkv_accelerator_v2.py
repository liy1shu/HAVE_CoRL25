# Use flash attention & cross attention

from tqdm import tqdm
import json
import os
import math
import torch
import wandb
import torch.nn as nn
import torch_geometric.data as tgd
import rpad.pyg.nets.pointnet2 as pnp
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
from flash_attn.bert_padding import unpad_input, pad_input

os.environ["NCCL_P2P_LEVEL"] = "NVL"

BATCH_SIZE = 32  # 6
PN_BSZ = 6   # 8
LEARNING_RATE = 1e-4
EPOCHS = 1000


class ActionDataset(Dataset):
    def __init__(self, subset, root_dir='~/datasets/unevenobject'):
        self.directory = os.path.join(root_dir, subset)
        indices_path = os.path.join(self.directory, 'indices.json')
        with open(indices_path, 'r') as f:
            indices = json.load(f)

        # Construct a (history, action_to_evaluate) pair
        self.sample_idx_to_data_idx = indices["data_idx"]
        self.sample_idx_to_traj_idx = indices["traj_idx"]
        self.sample_idx_to_action_idx = indices["action_idx"]

    def __len__(self):
        return len(self.sample_idx_to_data_idx)

    def __getitem__(self, idx):
        data_idx = self.sample_idx_to_data_idx[idx]
        with open(os.path.join(self.directory, f'{data_idx}.pkl'), 'rb') as f:
            sample = pkl.load(f)
        traj_idx = self.sample_idx_to_traj_idx[idx]
        sample = sample[traj_idx]
        action_idx = self.sample_idx_to_action_idx[idx]

        action_to_evaluate_pcd = sample["evaluate"]["pcds"][action_idx]
        action_to_evaluate_flow = sample["evaluate"]["flows"][action_idx]

        max_gt_score = np.max(sample["evaluate"]["scores"])
        if max_gt_score < 0.1:
            max_gt_score = 0.1
        gt_scores = sample["evaluate"]["scores"][action_idx]

        action_pcds = sample["pcds"][:-1]
        action_flows = sample["flows"][:-1]
        action_results = sample["obs_flows"][:-1]
        action_scores = sample["scores"][:-1]   # bad history: -1, good history: 1
        weight = 1 # self.obj_id_to_weight[sample["obj_id"]]
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
    

    return action_to_evaluate, padded_action_pcds, padded_action_flows, padded_action_results, gt_scores, history_scores, padding_masks, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', torch.zeros(max_len, d_model))
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]


class FlashCrossMHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.flash_cross_attn = FlashCrossAttention(softmax_scale=self.scaling, attention_dropout=dropout)
        
    def forward(self, q, kv, key_padding_mask=None):
        batch_size, tgt_len, _ = q.shape
        src_len = kv.shape[1]
        
        # Project queries, keys, and values
        q = self.q_proj(q).view(batch_size, tgt_len, self.nhead, self.head_dim)
        k = self.k_proj(kv).view(batch_size, src_len, self.nhead, self.head_dim)
        v = self.v_proj(kv).view(batch_size, src_len, self.nhead, self.head_dim)
        
        # Handle padding if present
        if key_padding_mask is not None:
            # Unpad the inputs
            q_unpad, q_indices, q_cu_seqlens, q_max_seqlen, q_used_seqlens = unpad_input(q, ~key_padding_mask[:, :tgt_len])
            kv_unpad, kv_indices, kv_cu_seqlens, kv_max_seqlen, kv_used_seqlens = unpad_input(kv, ~key_padding_mask)
            
            # Reshape k and v and combine them
            k_unpad = self.k_proj(kv_unpad)
            v_unpad = self.v_proj(kv_unpad)
            kv_unpad = torch.stack([k_unpad, v_unpad], dim=2)  # Shape: [num_tokens, 2, nhead, head_dim]
            
            # Convert to half precision as required by flash attention
            q_unpad = q_unpad.half()
            kv_unpad = kv_unpad.half()
            
            # Call flash attention with unpadded inputs
            context = self.flash_cross_attn(
                q_unpad.view(-1, self.nhead, self.head_dim),  # [num_tokens, nhead, head_dim]
                kv_unpad.view(-1, 2, self.nhead, self.head_dim),  # [num_tokens, 2, nhead, head_dim]
                cu_seqlens=q_cu_seqlens,
                max_seqlen=q_max_seqlen,
                cu_seqlens_k=kv_cu_seqlens,
                max_seqlen_k=kv_max_seqlen
            )
            
            # Pad the output back
            context = pad_input(context, q_indices, batch_size, tgt_len)
        else:
            # If no padding, reshape q and combine k,v
            q = q.view(batch_size, tgt_len, self.nhead, self.head_dim).half()
            kv = torch.stack([k, v], dim=2).half()  # [batch_size, src_len, 2, nhead, head_dim]
            
            context = self.flash_cross_attn(
                q,  # [batch_size, tgt_len, nhead, head_dim]
                kv  # [batch_size, src_len, 2, nhead, head_dim]
            )
        
        # Project output
        output = self.out_proj(context.float().view(batch_size, tgt_len, self.d_model))
        return output


class SequentialEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Self-attention layers for history processing
        self.history_layers = nn.ModuleList([
            FlashCrossMHA(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.history_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.history_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, action_results, padding_mask=None):
        """
        # query: [batch_size, 1, d_model]  # Your query embedding
        history: [batch_size, seq_len, d_model]  # Your history embeddings
        padding_mask: [batch_size, seq_len]  # Mask for padding in history
        """
        # Process history with self-attention
        x_history = action_results
        for layer, norm, ffn in zip(self.history_layers, self.history_norms, self.history_ffns):
            # Self attention on history
            attended = layer(x_history, x_history, padding_mask)
            x_history = x_history + attended
            x_history = norm(x_history)
            
            # FFN
            x_history = x_history + ffn(x_history)
            x_history = norm(x_history)
        
        # Process query with cross-attention to processed history
        # history_scores = self.history_scorer(x_history)
        # return history_scores
            
        return x_history


class ActionProposalScorer(nn.Module):
    def __init__(self, embed_dim_q, momentum=0.99):
        """
        embed_dim_q: Dimension of the query embeddings
        momentum: EMA momentum for updating fallback score
        """
        super().__init__()

        self.dim = embed_dim_q
        self.momentum = momentum  # EMA decay factor

        # self.q_proj = nn.Linear(embed_dim_q, embed_dim_q)  
        # self.k_proj = nn.Linear(embed_dim_q, embed_dim_q)  
        self.v_proj = nn.Linear(embed_dim_q, embed_dim_q)  
        self.fallback_value_encoder = nn.Linear(embed_dim_q, embed_dim_q)

        self.final_scorer = nn.Sequential(
            nn.Linear(embed_dim_q, embed_dim_q),
            nn.ReLU(),
            nn.Linear(embed_dim_q, embed_dim_q // 2),
            nn.ReLU(),
            nn.Linear(embed_dim_q // 2, 1),
            nn.Tanh()
        )
        self.test = False

        # Running statistics for QK max and min
        self.register_buffer("qk_max", torch.tensor(-float("inf")))
        self.register_buffer("qk_min", torch.tensor(float("inf")))
        self.register_buffer("fallback_score", torch.tensor(0.0))

    def update_fallback_score(self, attn_scores):
        """ Updates fallback_score using EMA on (max + min) / 2. """
        batch_max = attn_scores.max().detach()
        batch_min = attn_scores.min().detach()
        new_qk_mean = (batch_max + batch_min) / 2

        # Exponential moving average update
        self.qk_max = self.momentum * self.qk_max + (1 - self.momentum) * batch_max
        self.qk_min = self.momentum * self.qk_min + (1 - self.momentum) * batch_min
        self.fallback_score = self.momentum * self.fallback_score + (1 - self.momentum) * new_qk_mean

    def forward(self, uncond_q, q, k, v, src_key_padding_mask=None):
        # q = self.q_proj(q)
        # k = self.k_proj(k)
        v = self.v_proj(v)
        # print(q.shape, uncond_q.shape)
        default_values = self.fallback_value_encoder(uncond_q).unsqueeze(1)

        # Compute QK scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (1 ** 0.5)  
        # Update fallback score using batch statistics
        if not self.test:
            if attn_scores.numel() > 0:
                self.update_fallback_score(attn_scores)

        # Apply padding mask
        if src_key_padding_mask is not None:
            mask_expanded = src_key_padding_mask.unsqueeze(1).expand(-1, q.shape[1], -1)
            attn_scores = attn_scores.masked_fill(mask_expanded, -1e9)

        # Use the updated fallback score
        fallback_scores = self.fallback_score.expand(q.shape[0], q.shape[1], 1)
        attn_scores = torch.cat([attn_scores, fallback_scores], dim=-1)

        attn_weights = F.softmax(attn_scores, dim=-1)

        v_padded = torch.cat([v, torch.ones(v.shape[0], 1, self.dim).to(v.device) * default_values], dim=-2)

        output = torch.matmul(attn_weights, v_padded)
        output = self.final_scorer(output)
        return output, self.final_scorer(default_values)


class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        self.query_result_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        self.action_encoder = pnp.PN2Encoder(3, d_model)
        self.motion_encoder = pnp.PN2Encoder(3, d_model)
        self.motion_sequence_encoder = SequentialEncoder(d_model, nhead, num_layers, dim_feedforward)
        self.motion_scorer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        self.action_sequence_encoder = SequentialEncoder(d_model, nhead, num_layers, dim_feedforward)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.scorer = ActionProposalScorer(d_model)
        
        self.d_model = d_model

    def translate_motion_to_pcd_features(self, action_pcds, action_results, src_key_padding_mask=None):
        motion_pcd_features = torch.zeros((action_results.shape[0], action_results.shape[1], self.d_model))
        
        features = None
        data_list = []
        # score_list = []
        for i in range(action_results.shape[0]):
            for j in range(action_results.shape[1]):
                if src_key_padding_mask[i, j + 1]:
                    break

                data_list.append(tgd.Data(
                    pos=action_pcds[i, j].float(),
                    x=action_results[i, j].float()
                ))

                if len(data_list) == PN_BSZ:
                    batch = tgd.Batch.from_data_list(data_list)
                    curr_features = self.motion_encoder(batch.to(action_results.device))
                    # curr_scores = self.motion_decoder(curr_features)
                    # score_list.append(curr_scores)
                    if features is None:
                        features = curr_features
                    else:
                        features = torch.concat([features, curr_features], dim=0)
                
                    data_list = []

        if len(data_list) != 0:
            # Calculate the last batch
            batch = tgd.Batch.from_data_list(data_list)
            curr_features = self.motion_encoder(batch.to(action_results.device))
            # curr_scores = self.motion_decoder(curr_features)
            # score_list.append(curr_scores)
            if features is None:
                features = curr_features
            else:
                features = torch.concat([features, curr_features], dim=0)

        ptr = 0
        for i in range(action_results.shape[0]):
            for j in range(action_results.shape[1]):
                if src_key_padding_mask[i,j + 1]:
                    break
                motion_pcd_features[i, j] = features[ptr]
                ptr += 1
                
        # Pass through transformer to get the scores.
        motion_pcd_features = self.pos_encoder(motion_pcd_features.to(action_results.device))
        action_results_features = self.motion_sequence_encoder(motion_pcd_features, src_key_padding_mask[:, 1:])

        motion_scores = self.motion_scorer(action_results_features)
        
        return action_results_features, motion_scores
    
    
    def translate_action_to_features(self, action_pcds, action_flows, action_to_evaluate_feature, src_key_padding_mask=None):
        # First batch the action_tokens as featured point clouds.
        # Encode them
        action_pcd_features = torch.zeros((action_pcds.shape[0], action_pcds.shape[1], self.d_model))
        
        features = None
        data_list = []
        for i in range(action_pcds.shape[0]):
            for j in range(action_pcds.shape[1]):
                if src_key_padding_mask[i, j + 1]:
                    break

                data_list.append(tgd.Data(
                    pos=action_pcds[i, j].float().cuda(),
                    x=action_flows[i, j].float().cuda()
                ))

                if len(data_list) == PN_BSZ:
                    batch = tgd.Batch.from_data_list(data_list)
                    curr_features = self.action_encoder(batch.to(action_pcds.device))
                    if features is None:
                        features = curr_features
                    else:
                        features = torch.concat([features, curr_features], dim=0)
                
                    data_list = []

        if len(data_list) != 0:
            # Calculate the last batch
            batch = tgd.Batch.from_data_list(data_list)
            curr_features = self.action_encoder(batch.to(action_pcds.device))
            if features is None:
                features = curr_features
            else:
                features = torch.concat([features, curr_features], dim=0)

        ptr = 0
        for i in range(action_pcds.shape[0]):
            for j in range(action_pcds.shape[1]):
                if src_key_padding_mask[i, j+1]:
                    break
                action_pcd_features[i, j] = features[ptr]
                ptr += 1

        action_pcd_features = torch.concat([action_to_evaluate_feature.unsqueeze(1), action_pcd_features.to(action_pcds.device)], dim=-2)
        action_pcd_features = self.pos_encoder(action_pcd_features)
        action_pcd_features = self.action_sequence_encoder(action_pcd_features, src_key_padding_mask)

        return action_pcd_features.to(action_pcds.device)


    def forward(self, action_to_evaluate, action_pcds, action_flows, action_results, src_key_padding_mask=None):
        # Get query embedding (Query)
        action_to_evaluate_feature = self.action_encoder(action_to_evaluate)   
        # Get history action embeddings  (Keys)
        action_features = self.translate_action_to_features(action_pcds, action_flows, action_to_evaluate_feature, src_key_padding_mask)
        action_to_evaluate_feature_temporal = action_features[:, 0, :]
        action_features = action_features[:, 1:, :]
        # action_features = self.pos_encoder(action_features)
        # Get history action result embeddings  (Values)
        motion_features, motion_scores = self.translate_motion_to_pcd_features(action_pcds, action_results, src_key_padding_mask)

        # Get score embedding through cross-attention
        output, default_scores = self.scorer(uncond_q=action_to_evaluate_feature, q=action_to_evaluate_feature_temporal.unsqueeze(1), k=action_features, v=motion_features, src_key_padding_mask=src_key_padding_mask[:, 1:])

        return output.squeeze(-1), default_scores.squeeze(-1), motion_scores
# model.load_state_dict(torch.load('/home/yishu/failure_recovery/notebooks/sequential_predictor/score_modelv6_obsflow.pth'))

def test(model, dataloader, accelerator, epoch_id=None):
    model.eval()
    pred_scores_all = []
    pred_default_scores_all = []
    total_loss = 0
    total_default_loss = 0
    total_weighted_loss = 0
    total_count = 0

    # Prepare for distributed evaluation
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        # Move batch to appropriate device using accelerator
        (
            action_to_evaluate,
            padded_action_pcds,
            padded_action_flows,
            padded_action_results,
            gt_scores,
            history_scores,
            padding_masks,
            weights
        ) = [t.to(accelerator.device) for t in batch]

        with torch.no_grad():
            # Forward pass
            outputs, default_scores, scores = model(
                action_to_evaluate=action_to_evaluate,
                action_pcds=padded_action_pcds.float(),
                action_flows=padded_action_flows.float(),
                action_results=padded_action_results.float(),
                src_key_padding_mask=padding_masks,
            )

            # Append predictions
            pred_scores_all.append(outputs.detach().cpu().numpy())
            pred_default_scores_all.append(default_scores.detach().cpu().numpy())

            # Compute loss
            weighted_batch_loss = nn.MSELoss()(outputs * weights[:, None], gt_scores.float().unsqueeze(1) * weights[:, None])
            batch_loss = nn.MSELoss()(outputs, gt_scores.float().unsqueeze(1))
            batch_default_loss = nn.MSELoss()(default_scores, gt_scores.float().unsqueeze(1))


            total_weighted_loss += weighted_batch_loss.item()
            total_loss += batch_loss.item()
            total_default_loss += batch_default_loss.item()
            total_count += 1

    # Synchronize loss across processes
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    total_default_loss_tensor = torch.tensor(total_default_loss, device=accelerator.device)
    total_weighted_loss_tensor = torch.tensor(total_weighted_loss, device=accelerator.device)
    total_count_tensor = torch.tensor(total_count, device=accelerator.device)

    total_loss = accelerator.gather(total_loss_tensor).sum().item()
    total_default_loss = accelerator.gather(total_default_loss_tensor).sum().item()
    total_weighted_loss = accelerator.gather(total_weighted_loss_tensor).sum().item()
    total_count = accelerator.gather(total_count_tensor).sum().item()

    # Compute average loss (only on main process)
    avg_loss = total_loss / total_count if total_count > 0 else 0
    avg_default_loss = total_default_loss / total_count if total_count > 0 else 0

    # avg_loss = total_loss / 21   
    avg_weighted_loss = total_weighted_loss / 21  # 21 Objects

    if accelerator.is_main_process:
        print(f"Epoch {epoch_id} - Test Loss: {avg_loss:.4f} - Weighted Loss: {avg_weighted_loss:.4f}")

    return avg_loss, avg_default_loss, avg_weighted_loss



if __name__ == '__main__':
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    accelerator.init_trackers(
        project_name="scoring", 
        config={"learning_rate": LEARNING_RATE},
        init_kwargs={"wandb": {"entity": "your entity name"}}
    )

    # Load datasets
    train_dataset = ActionDataset("toy")
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    val_dataset = ActionDataset("toy")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    test_dataset = ActionDataset("toy")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


    device = accelerator.device

    model = TransformerModel(d_model=128, nhead=4, num_layers=4, dim_feedforward=256, max_len=200)
    model = model.to(device)
    # state_dict = torch.load("path to the state_dict", map_location=accelerator.device)
    # new_state_dict = {key[7:]: value for key, value in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the state_dict into the model

    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader, test_dataloader)

    global_step = 0
    best_loss = 1000
    best_mse_val = 100
    best_weighted_mse_val = 100
    best_mse_test = 100

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        running_loss, running_his_loss, running_pred_loss, running_default_pred_loss = 0.0, 0.0, 0.0, 0.0

        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")

        # Perform validation and testing every 5 epochs
        
        for batch in progress_bar:
            action_to_evaluate, padded_action_pcds, padded_action_flows, padded_action_results, gt_scores, history_scores, padding_masks, weights = [
                t.to(device) for t in batch
            ]

            optimizer.zero_grad()

            # Forward pass
            outputs, default_scores, scores = model(
                action_to_evaluate=action_to_evaluate,
                action_pcds=padded_action_pcds.float(),
                action_flows=padded_action_flows.float(),
                action_results=padded_action_results.float(),
                src_key_padding_mask=padding_masks
            )

            # Compute losses
            pred_loss = nn.MSELoss()(outputs, gt_scores.float().unsqueeze(1))
            default_pred_loss = nn.MSELoss()(default_scores, gt_scores.float().unsqueeze(1))
            
            his_score_loss = nn.MSELoss()(scores[~padding_masks[:, 1:]], history_scores.float().unsqueeze(1)) if scores is not None else 0.0

            loss = pred_loss + default_pred_loss + his_score_loss # Modify if you want to include history score loss

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            running_default_pred_loss += 0.5 * default_pred_loss.item()
            running_his_loss += his_score_loss.item() if isinstance(his_score_loss, torch.Tensor) else 0.0
            running_pred_loss += pred_loss.item()

            accelerator.log({
                "loss": loss,
                "default_score_loss": default_pred_loss, 
                "his_score": his_score_loss,
                "pred_score_loss": pred_loss
            }, step=global_step)
            global_step += 1


        if accelerator.is_main_process:
            # Log epoch-level metrics
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
            accelerator.save(model.state_dict(), "~/uneven-object-go/notebooks/0425toy/random_wizhis_supervision_v2_qkv_score_module_v6-2_last.pth")

            if (epoch+1) % 20 == 0:
                accelerator.save(model.state_dict(), f"~/uneven-object-go/notebooks/0425toy/random_wizhis_supervision_v2_qkv_score_module_v6-2_{epoch}.pth")

        if epoch % 1 == 0:
            # Validation
            # model.module.scorer.test=True
            mse_val, mse_default_val, weighted_mse_val = test(model, accelerator=accelerator, dataloader=val_dataloader)
            # mse_test, mse_default_test, weighted_mse_test = test(model, accelerator=accelerator, dataloader=test_dataloader)
            if accelerator.is_main_process:
                accelerator.log({"val_MSE": mse_val, "default_val_MSE": mse_default_val,"weighted_val_MSE": weighted_mse_val}, step=global_step)
                if mse_val < best_mse_val:
                    best_mse_val = mse_val
                    accelerator.save(model.state_dict(), "~/uneven-object-go/notebooks/0425toy/random_wizhis_supervision_v2_qkv_score_module_v6-2_valbest.pth")

                if weighted_mse_val < best_weighted_mse_val:
                    best_weighted_mse_val = weighted_mse_val
                    accelerator.save(model.state_dict(), "~/uneven-object-go/notebooks/0425toy/random_wizhis_supervision_v2_qkv_score_module_v6-2_weighted_valbest.pth")


                # accelerator.log({"test_MSE": mse_test, "default_test_MSE": mse_default_test, "weighted_test_MSE": weighted_mse_test}, step=global_step)
                # if mse_test < best_mse_test:
                #     best_mse_test = mse_test
                #     accelerator.save(model.state_dict(), "./ckpts/scoring_module/qkv_score_module_v6-2_testbest.pth")


            # model.module.scorer.test=False

