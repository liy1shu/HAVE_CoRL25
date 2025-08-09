# Use flash attention & cross attention
# TODO: fix...
# TODO: merge uneven object verifier code
from tqdm import tqdm
import json
import os
import math
import torch
import wandb
import torch.nn as nn
import torch_geometric.data as tgd
import rpad.pyg.nets.pointnet2 as pnp
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from have.verifier.datasets.action_dataset import ActionDataset, collate_fn
from have.verifier.models.ha_verifier import HAVErifier

os.environ["NCCL_P2P_LEVEL"] = "NVL" # For better multi-GPU communication


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

    LEARNING_RATE = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 10  # 6

    # breakpoint()
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    accelerator.init_trackers(
        project_name="failure_recovery", 
        config={"learning_rate": LEARNING_RATE},
        init_kwargs={"wandb": {"entity": "r-pad"}}
    )

    # Load datasets
    train_dataset = ActionDataset("train")
    # train_dataset_random = ActionDataset("train", "/project_data/held/yishul/datasets/failure_dataset/articulated_object_random/train_val/")
    # print(len(train_dataset), len(train_dataset_random))
    # breakpoint()
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)

    val_dataset = ActionDataset("val")
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    test_dataset = ActionDataset("val")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


    device = accelerator.device

    # Final model size
    model = HAVErifier(d_model=128, nhead=4, num_layers=4, dim_feedforward=256, max_len=200)

    # # Smaller model size
    # model = HAVErifier(d_model=64, nhead=2, num_layers=2, dim_feedforward=256, max_len=200)

    # # Bigger model size
    # model = HAVErifier(d_model=256, nhead=8, num_layers=6, dim_feedforward=512, max_len=200)

    model = model.to(device)
    # state_dict = torch.load("./ckpts/scoring_module/uncond_v2_qkv_score_module_v6-2_last.pth", map_location=accelerator.device)
    # new_state_dict = {key[7:]: value for key, value in state_dict.items()}
    # model.load_state_dict(new_state_dict)  # Load the state_dict into the model

    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, dataloader, val_dataloader, test_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader, test_dataloader)

    global_step = 0
    best_loss = 1000
    best_mse_val = 100
    best_weighted_mse_val = 100
    best_mse_test = 100

    # mse_val = test(model, accelerator=accelerator, dataloader=val_dataloader)

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
            # print(outputs.shape, default_scores.shape)
            his_score_loss = nn.MSELoss()(scores[~padding_masks[:, 1:]], history_scores.float().unsqueeze(1)) if scores is not None else 0.0

            loss = pred_loss + his_score_loss + default_pred_loss# Modify if you want to include history score loss

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
            # if epoch_loss < best_loss:
            #     # Save best model
            #     accelerator.save(model.state_dict(), "./ckpts/scoring_module/trainval_score_module_v6-2_best.pth")
            #     best_loss = epoch_loss
            # accelerator.save(model.state_dict(), "./ckpts/scoring_module/random_nodefault_supervision_v2_qkv_score_module_v6-2_last.pth")

            accelerator.save(model.state_dict(), "./ckpts/scoring_module/rebuttal/tvu_bigger_v2_qkv_score_module_v6-2_last.pth") #smaller


        if epoch % 1 == 0:
            # Validation
            model.module.scorer.test=True
            mse_val, mse_default_val, weighted_mse_val = test(model, accelerator=accelerator, dataloader=val_dataloader)
            # mse_test, mse_default_test, weighted_mse_test = test(model, accelerator=accelerator, dataloader=test_dataloader)
            if accelerator.is_main_process:
                accelerator.log({"val_MSE": mse_val, "default_val_MSE": mse_default_val,"weighted_val_MSE": weighted_mse_val}, step=global_step)
                if mse_val < best_mse_val:
                    best_mse_val = mse_val
                    # accelerator.save(model.state_dict(), "./ckpts/scoring_module/random_nodefault_supervision_v2_qkv_score_module_v6-2_valbest.pth")
                    accelerator.save(model.state_dict(), "./ckpts/scoring_module/rebuttal/tvu_bigger_v2_qkv_score_module_v6-2_last.pth")

                if weighted_mse_val < best_weighted_mse_val:
                    best_weighted_mse_val = weighted_mse_val
                    # accelerator.save(model.state_dict(), "./ckpts/scoring_module/random_nodefault_supervision_v2_qkv_score_module_v6-2_weighted_valbest.pth")
                    accelerator.save(model.state_dict(), "./ckpts/scoring_module/rebuttal/tvu_bigger_v2_qkv_score_module_v6-2_last.pth")


                # accelerator.log({"test_MSE": mse_test, "default_test_MSE": mse_default_test, "weighted_test_MSE": weighted_mse_test}, step=global_step)
                # if mse_test < best_mse_test:
                #     best_mse_test = mse_test
                #     accelerator.save(model.state_dict(), "./ckpts/scoring_module/qkv_score_module_v6-2_testbest.pth")


            model.module.scorer.test=False

