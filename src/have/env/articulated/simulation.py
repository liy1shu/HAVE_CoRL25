# TODO: clean and migrate the simulation codes here


# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os
import sys
import numpy as np
import pybullet as p
import torch
from pathlib import Path
import hydra
import rpad.partnet_mobility_utils.articulate as pma
from have.env.articulated.simulation import *
from have.env.articulated.suction import *
from have.env.articulated.suction import GTFlowModel, PMSuctionSim
from have.utils.tracker import Tracker
from have.generator.metrics.trajectory import flow_metrics
from rpad.partnet_mobility_utils.data import PMObject
import torch_geometric.data as tgd

from have.env.articulated.suction import (  # compute_flow,; run_trial_with_history,
    GTFlowModel,
    PMSuctionSim,
    point_direction_to_grasp_field,
    flow_to_grasp_field,
    choose_grasp_points
)


def simulation_articulated(obj_id, sampling_model, model, max_step=10, bsz=30, tracking=False, score_filter=True, oracle_sampler=False, oracle_score=False, grasp_selection=False, joint_id=0, camera_pos=[-2, 0, 2], multimodal_door=False, device=None):
    lev_diff_thres = 0.2
    step_to_open = 31
    if oracle_sampler and oracle_score:
        bsz = 1

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
        project_root = hydra.utils.get_original_cwd()
        delta_absolute_path = os.path.join(project_root, 'src/have/utils/DELTA')
        if delta_absolute_path is not None and delta_absolute_path not in sys.path:
            sys.path.insert(0, delta_absolute_path)

        delta_track_ckpt = os.path.join(delta_absolute_path, "checkpoints/densetrack3d.pth")
        delta_tracker = Tracker(env.render_env.camera, delta_absolute_path, delta_track_ckpt)

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




if __name__ == "__main__":
    import random
    np.random.seed(2003)
    torch.manual_seed(2003)
    # trial_flow(obj_id="41083", available_joints=["link_0"], gui=True, website=False)
    # trial_gt_trajectory(obj_id="8877", traj_len=3, available_joints=['link_2'], gui=False, website=True)
    # breakpoint()
    # trial_with_prediction(obj_id="35059", traj_len=15, n_step=1, gui=True)

    # length = 15
    # network_15 = create_network(
    #     traj_len=15,
    #     ckpt_file="/home/yishu/failure_recovery/scripts/logs/train_flowbot/2023-07-19/14-51-22/checkpoints/epoch=94-step=74670-val_loss=0.00-weights-only.ckpt",
    # )

    # # length = 1
    # network_1 = pnp.PN2Dense(
    #     in_channels=0,
    #     out_channels=3,
    #     p=pnp.PN2DenseParams(),
    # )
    # ckpt = torch.load("/home/yishu/failure_recovery/pretrained/fullset_half_half_flowbot.ckpt")
    # network_1.load_state_dict(
    #     {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    # )
    # network_1.eval()
    # # network_1.load_state_dict(torch.load()["state_dict"])
    # trial_figs, trial_results, sim_trajectory = trial_with_prediction(
    #     obj_id="102358", network=network_1, n_step=30, gui=False, website=True, all_joint=True
    # )
    # print(trial_results)

    # figs[list(figs.keys())[0]].show()
    # trial_with_prediction(obj_id="35059", network=network_15, n_step=1, gui=False, all_joint=False)

    # # Trial with dit
    # from have.generator.models.modules.dit_models import DiT

    # torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    # np.set_printoptions(precision=10)

    # network = DiT(
    #     in_channels=3 + 3,
    #     depth=5,
    #     hidden_size=128,
    #     num_heads=4,
    #     # depth=12,
    #     # hidden_size=384,
    #     # num_heads=6,
    #     learn_sigma=True,
    # ).cuda()
    # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_diffuser_dit/2024-03-30/07-12-41/checkpoints/epoch=359-step=199080-val_loss=0.00-weights-only.ckpt"
    # # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_diffuser_dit/2024-05-02/12-35-27/checkpoints/epoch=109-step=243100-val_loss=0.00-weights-only.ckpt"
    from hydra import compose, initialize

    initialize(config_path="../../../configs", version_base="1.3")
    cfg = compose(config_name="eval_sim")

    # from have.generator.models.flow_diffuser_dit import (
    #     FlowTrajectoryDiffuserSimulationModule_DiT,
    # )

    # model = FlowTrajectoryDiffuserSimulationModule_DiT(
    #     network, inference_cfg=cfg.inference, model_cfg=cfg.model
    # ).cuda()
    # model.load_from_ckpt(ckpt_file)
    # model.eval()

    from have.generator.models.flow_diffuser_hispndit import (
        FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
    )
    from have.generator.models.modules.dit_models import PN2HisDiT
    from have.generator.models.modules.history_encoder import HistoryEncoder

    # History model
    network = {
        "DiT": PN2HisDiT(
            history_embed_dim=128,
            in_channels=3,
            depth=5,
            hidden_size=128,
            num_heads=4,
            # depth=8,
            # hidden_size=256,
            # num_heads=4,
            learn_sigma=True,
        ).cuda(),
        "History": HistoryEncoder(
            history_dim=128,
            history_len=1,
            batch_norm=True,
            transformer=False,
            repeat_dim=False,
        ).cuda(),
    }

    # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_diffuser_hisdit/2024-05-10/12-09-08/checkpoints/epoch=439-step=243320-val_loss=0.00-weights-only.ckpt"
    ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_diffuser_hispndit/2024-05-25/02-00-54/checkpoints/epoch=299-step=248700-val_loss=0.00-weights-only-backup.ckpt"
    history_model = FlowTrajectoryDiffuserSimulationModule_HisPNDiT(
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    history_model.load_from_ckpt(ckpt_file)
    history_model.eval()

    # import rpad.pyg.nets.pointnet2 as pnp_orig

    # from have.generator.models.flow_trajectory_predictor import (
    #     FlowSimulationInferenceModule,
    # )

    # network = pnp_orig.PN2Dense(
    #     in_channels=0,
    #     out_channels=3,
    #     p=pnp_orig.PN2DenseParams(),
    # )#.cuda()
    # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_pn++/2024-05-26/02-37-08/checkpoints/epoch=98-step=109395-val_loss=0.00-weights-only.ckpt"
    # ckpt = torch.load(ckpt_file)
    # model = FlowSimulationInferenceModule(
    #     network, cfg.inference, cfg.model
    # )

    # trial_figs, trial_results, sim_trajectory = trial_with_prediction(
    #     obj_id="8877", network=model, n_step=30, gui=False, website=True, available_joints=["link_1"], all_joint=False
    # )
    # breakpoint()

    # switch_model = FlowSimulationInferenceModule(
    #     network, cfg.switch_inference, cfg.switch_model
    # ).cuda()
    # # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_pn++/2024-03-30/08-16-05/checkpoints/epoch=88-step=98345-val_loss=0.00-weights-only.ckpt"
    # # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_pn++/2024-05-25/04-17-41/checkpoints/epoch=95-step=53088-val_loss=0.00-weights-only.ckpt"
    # ckpt_file = "/home/yishu/failure_recovery/logs/train_trajectory_pn++/2024-05-26/02-37-08/checkpoints/epoch=98-step=109395-val_loss=0.00-weights-only.ckpt"
    # switch_model.load_from_ckpt(ckpt_file)
    # switch_model.eval()

    obj_id = "8877"  # 8877
    # trial_figs, trial_results, sim_trajectory = trial_with_diffuser(
    # trial_figs, trial_results, sim_trajectory = trial_with_switch_models(
    trial_figs, trial_results, sim_trajectory = trial_with_diffuser_history(
        # obj_id="8877",
        obj_id=obj_id,
        # model=model,
        # switch_model=switch_model,
        model=history_model,
        history_model=history_model,
        # history_for_models=[False, False],
        n_step=30,
        gui=False,
        website=cfg.website,
        all_joint=False,
        available_joints=["link_1"],
        # return_switch_ids=True,
    )

    # x = [i for i in range(31)]
    # y, colors = sim_trajectory[0]
    # colors = ["red" if color else "blue" for color in colors[1:]]

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 6))
    # for i in range(len(x) - 1):
    #     plt.plot(x[i : i + 2], y[i : i + 2], color=colors[i])

    # plt.xlabel("Step")
    # plt.yticks(np.linspace(0, 1, 11))
    # plt.ylabel("Open ratio")
    # plt.title(f"DiT & FowBot - Door {obj_id}")
    # plt.savefig(
    #     f"/home/yishu/failure_recovery/notebooks/analysis/traj_visuals/{obj_id}_dit&flowbot.jpg"
    # )
    breakpoint()
