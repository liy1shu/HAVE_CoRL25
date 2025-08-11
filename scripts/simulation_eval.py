# TODO: fix
# TODO: merge uneven object dataset collection code
import random
import json
import os
from tqdm import tqdm
import torch
import torch_geometric.data as tgd
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from src.scoring_model import TransformerModel  # Original model
# from src.scoring_model_cross import TransformerModel  # With cross attention
# from src.scoring_model_qkv import TransformerModel   #With qkv
from src.scoring_model_qkv_v2 import TransformerModel
from have.generator.metrics.trajectory import flow_metrics

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import lightning as L
import numpy as np
# Since most of us are training on 3090s+, we can use mixed precision.
torch.set_float32_matmul_precision("highest")

# Global seed for reproducibility.
L.seed_everything(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(d_model=128, nhead=4, num_layers=4, dim_feedforward=256, max_len=200)
model = model.to(device)


class InferenceConfig:
    def __init__(self):
        self.batch_size = 1
        self.trajectory_len = 1
        self.mask_input_channel = False

inference_config = InferenceConfig()

class ModelConfig:
    def __init__(self):
        self.num_train_timesteps = 100
        self.predict_xstart = False
        self.predict_v = False

model_config = ModelConfig()

from have.generator.models.flow_diffuser_pndit import FlowTrajectoryDiffuserInferenceModule_PNDiT
from have.generator.nets.dit_models import PN2DiT

network = PN2DiT(
    in_channels=3,
    depth=5,
    hidden_size=128,
    patch_size=1,
    num_heads=4,
    n_points=1200,
)

sampling_model = FlowTrajectoryDiffuserInferenceModule_PNDiT(network, inference_cfg=inference_config, model_cfg=model_config)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sampling_model = sampling_model.to(device)


import rpad.partnet_mobility_utils.articulate as pma
from have.env.articulated.simulation.simulation import *
from have.env.articulated.simulation.suction import *
from have.env.articulated.simulation.suction import GTFlowModel, PMSuctionSim
from tracking.delta_tracker import Tracker


def point_direction_to_grasp_field(P_world, link_ixs, best_point, best_flow, grasp_selection=False, normalize=False):
    if torch.is_tensor(best_flow):
        best_flow_np = best_flow.numpy()
    else:
        best_flow_np = best_flow
    if normalize:
        best_flow_np = best_flow_np / np.linalg.norm(best_flow_np)
    grasp_flow_field = np.zeros_like(P_world) + best_flow_np[np.newaxis, :]
    pcd_dist = np.power(P_world - best_point, 2).sum(axis=-1)
    if grasp_selection:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
    else:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]  # TODO: change to -1
    grasp_flow_field = grasp_flow_field * coeff
    grasp_flow_field *= link_ixs[:, np.newaxis]
    return grasp_flow_field


def flow_to_grasp_field(pred_flow, P_world, link_ixs=None, grasp_selection=False):
    norms = np.linalg.norm(pred_flow[link_ixs], axis=-1)
    grasp_point_idx = np.argmax(norms)
    best_flow = pred_flow[link_ixs][grasp_point_idx].numpy()
    best_point = P_world[link_ixs][grasp_point_idx]
    best_flow /= np.linalg.norm(best_flow)
    grasp_flow_field = np.zeros_like(P_world) + best_flow[np.newaxis, :]
    pcd_dist = np.power(P_world - best_point, 2).sum(axis=-1)
    if grasp_selection:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
    else:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]   # TODO: change to -1 
    grasp_flow_field = grasp_flow_field * coeff
    if link_ixs is not None:
        grasp_flow_field *= link_ixs[:, np.newaxis]
    return grasp_flow_field


def simulation_grasp(obj_id, sampling_model, model, max_step=10, bsz=30, tracking=False, score_filter=True, oracle_sampler=False, oracle_score=False, grasp_selection=False, joint_id=0, camera_pos=[-2, 0, 2], multimodal_door=False):
    lev_diff_thres = 0.2
    step_to_open = 31
    if oracle_sampler and oracle_score:
        bsz = 1

    # pm_dir = os.path.expanduser("/home/yishu/failure_recovery/local_workspace/objects")
    if multimodal_door:
        pm_dir = os.path.expanduser("/home/yishu/datasets/failure_history_door/raw")
    else:
        pm_dir = os.path.expanduser("/home/yishu/datasets/partnet-mobility/raw")
    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type("slider")
    available_joints = [joint.name for joint in available_joints]
    if joint_id is int:
        target_link = available_joints[joint_id]
    else:
        target_link = joint_id

    env = PMSuctionSim(obj_id, pm_dir, gui=False, camera_pos=camera_pos)
    gt_model = GTFlowModel(raw_data, env)

    if tracking: 
        delta_tracker = Tracker(env)

    env.disable_self_collision()
    for link_to_disable_collision in [joint.name for joint in raw_data.semantics.sems]:
        if link_to_disable_collision != target_link:
            env.disable_collision(env.render_env.link_name_to_index[link_to_disable_collision])
        else:
            env.disable_collision(env.render_env.link_name_to_index[link_to_disable_collision], body=False, floor=True)

    # Close all joints:
    for link_to_restore in [
        joint.name
        for joint in raw_data.semantics.by_type("hinge")
        + raw_data.semantics.by_type("slider")
    ]:
        info = p.getJointInfo(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[link_to_restore],
            env.render_env.client_id,
        )
        init_angle, target_angle = info[8], info[9]
        env.set_joint_state(link_to_restore, init_angle)


    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    history_pcds = []
    history_flows = []
    history_results = []   # Get the obs flow (This currently still requires ground truth)
    animation = FlowNetAnimation()


    all_step_logs = {
        "intermediates": [],
        "preds": [],
        "mask": [],
        "scores": [],
        "P_worlds": [],
        "gt_flow": [],
    }

    last_step_grasp_point = None  # Record the last grasp point - for sgp policy

    step_cnt_with_regrasp = 0

    for step_id in range(max_step):
        curr_joint_angles = env.render_env.get_joint_angles()
        curr_angle = env.get_joint_value(target_link)
        curr_rel_angle = (curr_angle - init_angle) / (target_angle - init_angle)
        # if curr_rel_angle < 0:
        #     env.set_joint_state(target_link, init_angle)
        #     print("Reset!", env.get_joint_value(target_link))
        if curr_rel_angle > 0.05 and step_to_open == 31:
            step_to_open = step_id + 1

        pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
        gt_flow = gt_model(pc_obs)
        all_step_logs["gt_flow"].append(gt_flow)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        
        if not link_ixs.any():  # No point segmented..
            link_ixs = np.logical_or(link_ixs, True)  # Take every point into consideration

        print("Current rel angle: ", curr_rel_angle)

        all_step_logs["P_worlds"].append(P_world)
        all_step_logs["mask"].append(link_ixs)

        if tracking:
            delta_tracker.append_observation(rgb.astype(np.float32), depth)
            if len(all_step_logs["P_worlds"]) != 1:  # Not the first step - we need to calculate flow
                dense_track_flows = delta_tracker.get_latest_obs_flow(all_step_logs["P_worlds"][-2])
                history_results.append(torch.from_numpy(dense_track_flows))

        if step_id == 0:
            flow_pred = sampling_model.predict(P_world)
            grasp_flow_field = flow_to_grasp_field(flow_pred.squeeze(1), P_world, link_ixs, grasp_selection=grasp_selection)

        elif not score_filter:  # No score_filter, but still has classifier guidance!
            
            flow_pred = sampling_model.predict(P_world, bsz=1) #.reshape(-1, 1200, 1, 3)
            grasp_flow_field = flow_to_grasp_field(flow_pred.squeeze(1), P_world, link_ixs, grasp_selection=grasp_selection)

            # all_step_logs["intermediates"].append(intermediates)
            all_step_logs["preds"].append(flow_pred)
            
        else:
            best_score = -10
            final_flow_pred = None
            final_grasp_flow_field = None

            with torch.no_grad():
                pred_flows = sampling_model.predict(P_world, bsz=bsz)
                pred_flows = pred_flows.reshape(-1, 1200, 1, 3)
                
                if oracle_sampler:  # Add in ground truth action 
                    bsz += 1
                    pred_flows = torch.concat([pred_flows, gt_flow.unsqueeze(0).unsqueeze(2)], dim=0)

                # all_step_logs["intermediates"].append(intermediates)
                all_step_logs["preds"].append(pred_flows)
                data_list = []
                grasp_flow_fields = []
                for pred_flow in pred_flows:

                    grasp_flow_field = flow_to_grasp_field(pred_flow.squeeze(1), P_world, link_ixs, grasp_selection=grasp_selection)
                    grasp_flow_fields.append(grasp_flow_field)

                    data_list.append(tgd.Data(
                        pos=torch.from_numpy(P_world).float(),
                        x=torch.from_numpy(grasp_flow_field).float()#.to(device)
                    ))
                
                if oracle_score:   # Use gt rmse as scores
                    rmses, _, _ = flow_metrics(
                        pred_flows.squeeze()[torch.from_numpy(link_ixs).unsqueeze(0).repeat(bsz, 1)].reshape(bsz, -1, 3), 
                        gt_flow.unsqueeze(0).repeat(bsz, 1, 1)[torch.from_numpy(link_ixs).unsqueeze(0).repeat(bsz, 1)].reshape(bsz, -1, 3), reduce=False
                    )
                    scores = (-1) * rmses.reshape(bsz, -1).mean(-1)

                else:  # Use scoring module as scores
                    actions = tgd.Batch.from_data_list(data_list)
                    print(len(history_pcds), len(history_flows), len(history_results), step_cnt_with_regrasp)
                    scores, default_scores, _ = model(
                        action_to_evaluate = actions.to(device), 
                        action_pcds = torch.stack(history_pcds, dim=0).unsqueeze(0).repeat(bsz, 1, 1, 1).float().to(device), 
                        action_flows = torch.stack(history_flows).unsqueeze(0).repeat(bsz, 1, 1, 1).float().to(device), 
                        action_results = torch.stack(history_results).unsqueeze(0).repeat(bsz, 1, 1, 1).float().to(device), 
                        src_key_padding_mask = torch.zeros(bsz, len(history_pcds) + 1).bool().to(device)
                    )
                    all_step_logs["scores"].append(scores)

                max_score_id = torch.argmax(scores)
                # assert max_score_id == bsz - 1, "? wtf just happened"
                flow_pred = pred_flows[max_score_id]
                grasp_flow_field = grasp_flow_fields[max_score_id]

        flow_pred = flow_pred.squeeze(1)
        max_contact_trial_id = 10

        # For sgp
        if last_step_grasp_point is not None:  # Still grasps!
            gripper_tip_pos, _ = p.getBasePositionAndOrientation(
                env.gripper.body_id
            )
            pcd_dist = torch.tensor(
                P_world[link_ixs] - np.array(gripper_tip_pos)
            ).norm(dim=-1)
            grasp_point_id = pcd_dist.argmin()
            lev_diff = best_flows.norm(dim=-1) - flow_pred[link_ixs][
                grasp_point_id
            ].norm(dim=-1)


        if last_step_grasp_point is None or lev_diff[0] > lev_diff_thres:  # Re-grasp!!!
            env.reset_gripper(target_link)
            p.stepSimulation(
                env.render_env.client_id
            )  # Make sure the constraint is lifted
            
            best_flow_ixs, best_flows, best_points = choose_grasp_points(
                flow_pred[link_ixs],  #torch.from_numpy(grasp_flow_field[link_ixs]), # 
                P_world[link_ixs], filter_edge=False, k=40
            )
            if not grasp_selection:
                best_flow_ix_id, contact = env.teleport(best_points, best_flows, target_link=target_link)
                step_cnt_with_regrasp += best_flow_ix_id + 1
                # Movement caused by contact process
                if not contact:
                    print("Cannot contact")
                    p.disconnect(env.render_env.client_id)
                    return [None, None, step_to_open], None, False, curr_rel_angle

                best_flow = flow_pred[link_ixs][best_flow_ixs[best_flow_ix_id]].numpy()
                # best_flow = grasp_flow_field[link_ixs][best_flow_ixs[best_flow_ix_id]]#.numpy()
                best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]
                env.attach()
                grasp_flow_field = point_direction_to_grasp_field(P_world, link_ixs, best_point, best_flow, grasp_selection=grasp_selection, normalize=True)
                history_pcds.append(torch.from_numpy(P_world))
                history_flows.append(torch.from_numpy(grasp_flow_field))
                animation.add_trace(
                    torch.as_tensor(P_world),
                    # torch.as_tensor([pcd[mask]]),
                    # torch.as_tensor([flow[mask]]),
                    torch.as_tensor([P_world]),
                    # torch.as_tensor([grasp_flow_field * 3]),
                    torch.as_tensor([grasp_flow_field * 3]),
                    "red",
                )

            else:
                contact = False
                max_contact_trial = 40
                contact_attempts = 0
                while not contact and contact_attempts < max_contact_trial:
                    # If grasp selection: execute contact, repredict scores for the contact points, and re-grasp
                    contact_bsz = len(best_flows)
                    contact_attempts += 1

                    data_list = []
                    for best_flow, best_point in zip(best_flows, best_points):
                        data_list.append(tgd.Data(
                            pos=torch.from_numpy(P_world).float().to(device),
                            x=torch.from_numpy(point_direction_to_grasp_field(P_world, link_ixs, best_point=best_point, best_flow=best_flow, grasp_selection=grasp_selection)).float().to(device)  # x[b_id]#
                        ))
                    actions = tgd.Batch.from_data_list(data_list)

                    if len(history_flows) == 0:
                        max_score_id = 0
                    else:
                        action_pcds = torch.stack(history_pcds, dim=0).unsqueeze(0).repeat(contact_bsz, 1, 1, 1).float().to(device)
                        # action_pcds.requires_grad_(True)
                        action_flows = torch.stack(history_flows).unsqueeze(0).repeat(contact_bsz, 1, 1, 1).float().to(device)
                        # action_flows.requires_grad_(True)
                        action_results = torch.stack(history_results).unsqueeze(0).repeat(contact_bsz, 1, 1, 1).float().to(device)
                        # action_results.requires_grad_(True)
                        src_key_padding_mask = torch.zeros(contact_bsz, len(history_pcds) + 1).float().to(device)  # step_cnt_with_grasps
                        # print(action_pcds.shape, action_flows.shape, action_results.shape, src_key_padding_mask.shape)
                        with torch.no_grad():
                            scores, default_scores, _ = model(
                                action_to_evaluate=actions,
                                action_pcds=action_pcds,
                                action_flows = action_flows, 
                                action_results = action_results, 
                                src_key_padding_mask = src_key_padding_mask.bool(),
                            )  # Shape: [batch_size]
                        max_score_id = torch.argmax(scores)
                    
                    grasp_point = best_points[max_score_id]
                    grasp_flow = best_flows[max_score_id]
                    grasp_point_ixs = best_flow_ixs[max_score_id]

                    grasp_flow_field = point_direction_to_grasp_field(P_world, link_ixs, grasp_point, grasp_flow, grasp_selection=grasp_selection, normalize=True)
                    history_pcds.append(torch.from_numpy(P_world))
                    history_flows.append(torch.from_numpy(grasp_flow_field))

                    animation.add_trace(
                        torch.as_tensor(P_world),
                        # torch.as_tensor([pcd[mask]]),
                        # torch.as_tensor([flow[mask]]),
                        torch.as_tensor([P_world]),
                        # torch.as_tensor([grasp_flow_field * 3]),
                        torch.as_tensor([grasp_flow_field * 3]),
                        "red",
                    )

                    # # Release the previous contact!
                    # env.reset_gripper(target_link)
                    # p.stepSimulation(
                    #     env.render_env.client_id
                    # )  # Make sure the constraint is lifted

                    best_flow_ix_id, contact = env.teleport(grasp_point[np.newaxis, ...], grasp_flow[np.newaxis, ...], target_link=target_link)
                    # Movement caused by contact process
                    contact_delta = env.get_joint_value(target_link) - curr_angle
                    print("Angle caused by contact: ", contact_delta)

                    step_cnt_with_regrasp += 1
                    if contact:
                        # print("Cannot contact")
                        # p.disconnect(env.render_env.client_id)
                        # return None, None, False
                        break

                    # Record the motion (observation flow)
                    history_results.append(torch.zeros_like(flow_pred))

                if contact:
                    best_flow = flow_pred[link_ixs][grasp_point_ixs].numpy()
                    best_point = P_world[link_ixs][grasp_point_ixs]
                    env.attach()
                else:
                    p.disconnect(env.render_env.client_id)
                    return [None, None, step_to_open], None, False, curr_rel_angle
                
        else:  # No grasping
            best_flow_ixs, best_flows, best_points = choose_grasp_points(
                flow_pred[link_ixs],  # torch.from_numpy(grasp_flow_field[link_ixs]), #
                P_world[link_ixs], filter_edge=False, k=1
            )
            best_flow = flow_pred[link_ixs][best_flow_ixs[0]].numpy()
            # best_flow = grasp_flow_field[link_ixs][best_flow_ixs[0]]#.numpy()
            best_point = P_world[link_ixs][grasp_point_id]

            grasp_flow_field = point_direction_to_grasp_field(P_world, link_ixs, best_point, best_flow, grasp_selection=grasp_selection)
            history_pcds.append(torch.from_numpy(P_world))
            history_flows.append(torch.from_numpy(grasp_flow_field))
            step_cnt_with_regrasp += 1
        
        last_step_grasp_point = best_point
        reset = env.pull_with_constraint(best_flow, target_link=target_link, n_steps=100, constraint=True)
        if reset:
            last_step_grasp_point = None

        if not tracking:
            # Record the motion (observation flow)
            P_world_new = pma.articulate_joint(
                raw_data,
                curr_joint_angles,
                target_link,
                env.get_joint_value(target_link) - curr_angle,  # Articulate by only a little bit.
                P_world,
                pc_seg,
                env.render_env.link_name_to_index,
                env.render_env.T_world_base,
            )
            obs_flow = P_world_new - P_world
            history_results.append(torch.from_numpy(obs_flow))

        # Check succeed or not
        curr_rel_angle = (env.get_joint_value(target_link) - init_angle) / (target_angle - init_angle)
        if curr_rel_angle > 0.9:
            print("Succeed!")
            all_step_logs["animation"] = animation
            p.disconnect(env.render_env.client_id)
            return [step_id + 1, step_cnt_with_regrasp, step_to_open], all_step_logs, True, 1
        # if curr_rel_angle < 0:
        #     env.set_joint_state(target_link, init_angle)
        #     print("Reset!", env.get_joint_value(target_link))

    all_step_logs["animation"] = animation

    p.disconnect(env.render_env.client_id)
    return [step_id + 1, step_cnt_with_regrasp, step_to_open], all_step_logs, False, curr_rel_angle



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--door', type=bool, default=False, help="Multimodal Door dataset (default False)")
    parser.add_argument('--tvu', type=bool, default=False, help="Train Val Unseen dataset.")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--score_ckpt', type=str, default=None)
    parser.add_argument('--track', type=bool, default=False, help="Use tracking (default False)")
    parser.add_argument('--sample', type=int, default=30, help="How many actions to sample per step")
    parser.add_argument('--oracle_sampler', type=bool, default=False, help="Enable or disable oracle_sampler (default: False).")
    parser.add_argument('--oracle_score', type=bool, default=False, help="Enable or disable oracle_score (default: False).")
    parser.add_argument('--score_filter', type=bool, default=False, help="Enable or disable score_filter (default: False).")
    parser.add_argument('--grasp_selection', type=bool, default=False, help="Enable or disable grasp_selection (default: False).")
    parser.add_argument('--trainset', type=bool, default=False, help="Test train set of this split.")
    parser.add_argument('--size', type=str, default=None, help="Test train set of this split.")

    # Parse arguments
    args = parser.parse_args()
    multimodal_door = args.door
    tvu = args.tvu   # Evaluating the train val unseen
    use_tracking = args.track
    sample_cnt = args.sample
    oracle_sampler = args.oracle_sampler
    oracle_score = args.oracle_score
    score_filter = args.score_filter
    grasp_selection = args.grasp_selection
    trainset = args.trainset
    print("Oracle Sampler:", args.oracle_sampler)
    print("Oracle Score:", args.oracle_score)
    print("Score Filter:", args.score_filter)
    print("Grasp Selection:", args.grasp_selection)
    
    if args.size:
        if args.size == "s":
            model = TransformerModel(d_model=64, nhead=2, num_layers=2, dim_feedforward=256, max_len=200)
        elif args.size == "l":
            model = TransformerModel(d_model=256, nhead=8, num_layers=6, dim_feedforward=512, max_len=200)
        else:
            raise ValueError("Invalid size argument. Use 'small (s)' or 'large (l)'.")
        model = model.to(device)



    # sampling_model.load_from_ckpt("/home/yishu/failure_recovery/logs/train_trajectory_diffuser_pndit/2024-12-16/11-35-06/checkpoints/epoch=2949-step=73750-val_loss=0.00-weights-only.ckpt")   # Door dataset
    if multimodal_door:
        # sampling_model.load_from_ckpt("/home/yishu/failure_recovery/logs/train_trajectory_diffuser_pndit/2025-03-11/15-42-09/checkpoints/epoch=399-step=335200-val_loss=0.00-weights-only.ckpt")
        sampling_model.load_from_ckpt("/home/yishu/failure_recovery/ckpts/fullset_half_half_pndit.ckpt")
    else:
        if tvu:
            sampling_model.load_from_ckpt("/home/yishu/failure_recovery/ckpts/trainvalunseen_half_half_pndit.ckpt")
        else:
            sampling_model.load_from_ckpt("/home/yishu/failure_recovery/ckpts/fullset_half_half_pndit.ckpt")
    sampling_model.eval()


    if args.score_ckpt:
        print("Using specified ckpt: ", args.score_ckpt)
        state_dict = torch.load(args.score_ckpt)
    else:
        if args.grasp_selection:
            state_dict = torch.load("/home/yishu/failure_recovery/sequential_predictor/ckpts/fullset_score_module_v6-2_valbest_grasp_fromscratch.pth")   # With grasp
        
        elif multimodal_door:
            # state_dict = torch.load("./door_ckpts/latest_door_random_qkv_score_module_v6-2_valbest.pth")
            # state_dict = torch.load("./ckpts/random_moredoors_qkv_score_module_v6-2_last.pth")
            # state_dict = torch.load("./ckpts/final_scoring/random_v2_cont_qkv_score_module_v6-2_valbest.pth")
            state_dict = torch.load("./ckpts/random_v2_moredoors_qkv_score_module_v6-2_valbest.pth")
        else:
            # state_dict = torch.load("./ckpts/fullset_score_module_v6-2_valbest.pth")  # No grasp, original dataset
            # Train_val dataset
            # state_dict = torch.load("./ckpts/trainval_score_module_v6-2_valbest_new.pth")
            # state_dict = torch.load("./ckpts/trainval_score_module_v6-2_valbest_weighted_eval.pth")
            # state_dict = torch.load("./ckpts/trainval_score_module_v6-2_valbest.pth")
            # state_dict = torch.load("./ckpts/trainval_score_module_v6-2_testbest.pth")
            # state_dict = torch.load("./ckpts/trainval_newsplit_score_module_v6-2_valbest.pth")
            # state_dict = torch.load("./ckpts/trainval_newsplit_score_module_v6-2_weighted_valbest.pth")
            # state_dict = torch.load("./ckpts/qkv_score_module_v6-2_weighted_valbest.pth")
            # state_dict = torch.load("./ckpts/random_moredoors_qkv_score_module_v6-2_weighted_valbest.pth")
            # state_dict = torch.load("./ckpts/random_moredoors_qkv_score_module_v6-2_valbest.pth")
            # state_dict = torch.load("./ckpts/random_moredoors_qkv_score_module_v6-2_last.pth")
            # state_dict = torch.load("./ckpts/uncond_moredoors_qkv_score_module_v6-2_cont_weighted_valbest.pth")
            if tvu:
                state_dict = torch.load("./ckpts/train-val-unseen-random_v2_qkv_score_module_v6-2_valbest.pth")
                print("Loading train val unseen ckpts!")
            else:
                state_dict = torch.load("./ckpts/random_v2_moredoors_qkv_score_module_v6-2_valbest.pth")
    state_dict = {key[7:]:value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)  # Fullset V6.2
    model.scorer.test = True
    model.eval()

    # oracle_sampler=True #True
    # oracle_score=True  # False
    # score_filter=True
    # grasp_selection=False  #True
    if tvu:
        # Full dataset simulation
        with open('/home/yishu/failure_recovery/scripts/movable_links_fullset_000_full.json', 'r') as f:
            movable_links = json.load(f)

        with open('/home/yishu/failure_recovery/scripts/train_val_unseen.json', 'r') as f:  # Train val unseen split
            actual_data_split = json.load(f)

        test_obj_ids = {}
        with open('/home/yishu/failure_recovery/scripts/umpnet_data_split_new.json', 'r') as f:
            data_split = json.load(f)

            for obj_cat in data_split['train'].keys():
                # train_obj_ids += data_split['train'][obj_id]['train']
                for obj_id in (data_split['train'][obj_cat]['test'] + data_split['train'][obj_cat]['train']):
                    if trainset:   # Evaluate on the train categories
                        if obj_id in actual_data_split['train-test']:
                            if f'val_{obj_cat}' not in test_obj_ids.keys():
                                test_obj_ids[f'val_{obj_cat}'] = []
                            test_obj_ids[f'val_{obj_cat}'].append(obj_id)
                    else:
                        if obj_id in actual_data_split['train-test']:
                            # if f'val_{obj_cat}' not in test_obj_ids.keys():
                            #     test_obj_ids[f'val_{obj_cat}'] = []
                            # test_obj_ids[f'val_{obj_cat}'].append(obj_id)
                            continue
                        elif obj_id in actual_data_split['test']:
                            if f'test_{obj_cat}' not in test_obj_ids.keys():
                                test_obj_ids[f'test_{obj_cat}'] = []
                            test_obj_ids[f'test_{obj_cat}'].append(obj_id)

    elif not multimodal_door:

        # Full dataset simulation
        with open('/home/yishu/failure_recovery/scripts/movable_links_fullset_000_full.json', 'r') as f:
            movable_links = json.load(f)

        with open('/home/yishu/failure_recovery/scripts/umpnet_data_split_new.json', 'r') as f:
            data_split = json.load(f)

        # train_obj_ids = []  # TODO: read from the list..
        test_obj_ids = {}

        for obj_cat in data_split['train'].keys():
            # train_obj_ids += data_split['train'][obj_id]['train']
            if trainset:
                test_obj_ids[obj_cat] = data_split['train'][obj_cat]['train']
            else:
                test_obj_ids[obj_cat] = data_split['train'][obj_cat]['test']
    else:
        # Use the multimodal door dataset!
        with open('/home/yishu/failure_recovery/scripts/multimodal_door.json', 'r') as f:
            door_split = json.load(f)

        test_obj_ids_list = door_split['test'] + door_split['train-train']
        test_obj_ids = {}
        movable_links = {}
        for id in test_obj_ids_list:
            obj_cat = id.split('_')[0]
            if obj_cat not in test_obj_ids.keys():
                test_obj_ids[obj_cat] = []
            movable_links[id] = ['_'.join(id.split('_')[1:3])]
            test_obj_ids[obj_cat].append(id)

        # breakpoint()

    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/results/grasp_fullset{"_selection" if grasp_selection else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/oracle_results/new_fullset_sim_results{"_score" if score_filter else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results/sim_newsplit_weighted_results{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results/sim_oldstructure_results{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results{"_door" if multimodal_door else ""}/uncond_sim_qkv_results{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results{"_door" if multimodal_door else ""}/random_v2_sim_qkv_results{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results{"_door" if multimodal_door else ""}/latest_v2-1_sim_qkv_results{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results{"_door" if multimodal_door else ""}/qkv_results{"_DELTA" if use_tracking else ""}{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    result_path = f'/home/yishu/failure_recovery/sequential_predictor/analysis_results/{args.exp_name}{sample_cnt}_{"tvu_" if tvu else ""}{"door_" if multimodal_door else ""}qkv_results{"_DELTA" if use_tracking else ""}{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    # result_path = f'/home/yishu/failure_recovery/sequential_predictor/correct_results/{args.exp_name}{"tvu_" if tvu else ""}{"door_" if multimodal_door else ""}qkv_results{"_DELTA" if use_tracking else ""}{"_cascore" if score_filter else ""}{"_selection" if grasp_selection else ""}{"_orasampler" if oracle_sampler else ""}{"_orascore" if oracle_score else ""}.json'
    
    print(result_path)
    # breakpoint()
    
    step_cnts = {}
    step_with_grasp_cnts = {}
    steps_to_open = {}
    normalized_distance = {}
    success = {}
    # fail_because_contact = {}
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            existing_json = json.load(f)
            mean_steps = existing_json["Mean step"]
            mean_step_grasps = existing_json["Mean step grasps"]
            mean_steps_to_open = existing_json["Mean step to open"]
            mean_steps_to_open_clean = existing_json["Mean step to open (Clean)"]
            mean_success = existing_json["Success Rate"]
            mean_norm_dist = existing_json["Norm Dist"]
            fail_because_contact = existing_json["Contact Failure"]
    else:
        mean_steps = {}
        mean_step_grasps = {}
        mean_steps_to_open = {}
        mean_steps_to_open_clean = {}
        mean_success = {}
        mean_norm_dist = {}
        fail_because_contact = {}

    for obj_cat in list(test_obj_ids.keys())[::-1]:
        # if obj_cat not in ["StorageFurniture"]:  # Skip
        #     continue
        if obj_cat in mean_steps.keys():
            continue
        # if obj_cat in ["Refrigerator", "Kettle", "Table", "Window", "StorageFurniture"]:  # Skip
        # if obj_cat not in ["Door", "FoldingChair", "Safe", "Phone", "TrashCan"]:  # Skip
        #     continue
    # for obj_cat in ["Oven", "Table"]:  # FOR ORACLE DEBUGGING!!!!!!!!!!!!!
        step_cnts[obj_cat] = []
        step_with_grasp_cnts[obj_cat] = []
        steps_to_open[obj_cat] = []
        success[obj_cat] = []
        normalized_distance[obj_cat] = []
        # animations[obj_cat] = []
        fail_because_contact[obj_cat] = 0
        print(obj_cat)
        for obj_id in tqdm(test_obj_ids[obj_cat]):
            for joint_id in movable_links[obj_id]:
                print(obj_id, joint_id)
                for i in range(5):   # Repeat for 5 times - as in flowbothd evaluation
                    # camera_pos = [-4, 0, 2] if obj_cat == "TrashCan" else [-2, 0, 2]
                    camera_pos = [-4, 0, 4]
                    step_cnt, animation, succeed, rel_angle = simulation_grasp(obj_id, sampling_model=sampling_model, model=model, max_step=30, bsz=sample_cnt, score_filter=score_filter, oracle_sampler=oracle_sampler, oracle_score=oracle_score, grasp_selection=grasp_selection, joint_id=joint_id, camera_pos=camera_pos, multimodal_door=multimodal_door)
                    if succeed:
                        step_cnts[obj_cat].append(step_cnt[0])
                        step_with_grasp_cnts[obj_cat].append(step_cnt[1])
                    steps_to_open[obj_cat].append(step_cnt[2])
                    success[obj_cat].append(int(succeed))
                    normalized_distance[obj_cat].append(rel_angle)
                    # animations[obj_cat].append(animation)
                    if step_cnt is None:   # Means fail because no contact!
                        fail_because_contact[obj_cat] += 1

        if len(step_cnts[obj_cat]) != 0:
            mean_steps[obj_cat] = np.mean(step_cnts[obj_cat])
            mean_step_grasps[obj_cat] = np.mean(step_with_grasp_cnts[obj_cat])
            mean_steps_to_open[obj_cat] = np.mean(steps_to_open[obj_cat])
            mean_steps_to_open_clean[obj_cat] = np.mean([step for step in steps_to_open[obj_cat] if step != 31])
        else:
            mean_steps[obj_cat] = 1000
            mean_step_grasps[obj_cat] = 1000
            mean_steps_to_open[obj_cat] = 1000
            mean_steps_to_open_clean[obj_cat] = 1000
            
        mean_success[obj_cat] = np.mean(success[obj_cat])
        mean_norm_dist[obj_cat] = np.mean(normalized_distance[obj_cat])
        if len(success[obj_cat]) != 0:
            fail_because_contact[obj_cat] /= len(success[obj_cat])

        # Save temporary results!
        with open(result_path, 'w') as f:
            json.dump({"Success Rate": mean_success, "Norm Dist": mean_norm_dist, "Mean step to open": mean_steps_to_open, "Mean step to open (Clean)": mean_steps_to_open_clean, "Mean step": mean_steps, "Mean step grasps": mean_step_grasps, "Contact Failure": fail_because_contact}, f)
    

    print(mean_steps)
    print(mean_step_grasps)
    print(mean_success)
    print(mean_norm_dist)
    print(fail_because_contact)
    # breakpoint()

