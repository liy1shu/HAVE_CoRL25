import random
import json
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import rpad.partnet_mobility_utils.articulate as pma
from have.env.articulated.simulation import *
from have.env.articulated.suction import *
from have.env.articulated.suction import GTFlowModel, PMSuctionSim
from have.env.uneven.scoop_env import ScoopEnv


np.random.seed(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)

# For curated test dataset generation (sanity check, not used in general dataset generation)
curate_names = {
    "9127_plll": ["pllr", "pshl", "pshr"],
    "9127_pllr": ["plll", "pshr", "pshl"],
    "9127_pshl": ["pshr", "plll", "pllr"],
    "9127_pshr": ["pshl", "pllr", "plll"]
}
obj_ids = ["9127_plll", "9127_pllr", "9127_pshl", "9127_pshr"]

closed_gt_rate = 0.2
open_gt_rate = 0.8

# Action, Action result (point cloud after action), Action result (Tracking flow), Score (% of gt)
def generate_trajectory(pm_dir, step_cnt=-1, obj_id=-1, joint_id=0, evaluate_random_action_cnt=10, curate=False, curate_id=None):
    long_history = np.random.randint(0, 2)
    if long_history == 1:
        step_cnt = np.random.randint(1, 31) if step_cnt == -1 else step_cnt # Randomly generate a history length (history + 1 curr step).
    else:
        step_cnt = np.random.randint(1, 10) if step_cnt == -1 else step_cnt# joint_id = 0
    # obj_id = obj_ids[np.random.randint(0, 4) if obj_id == -1 else obj_id]

    animation = FlowNetAnimation()
    raw_data = PMObject(os.path.join(pm_dir, obj_id))
    available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type("slider")
    available_joints = [joint.name for joint in available_joints]
    target_link = joint_id #available_joints[joint_id]

    env = PMSuctionSim(obj_id, pm_dir, gui=False, camera_pos=[-4, 0, 4], camera_pos_random=True)
    gt_model = GTFlowModel(raw_data, env)
    env.reset()

    env.disable_self_collision()
    for link_to_disable_collision in [joint.name for joint in raw_data.semantics.sems]:
        if link_to_disable_collision != target_link:
            env.disable_collision(env.render_env.link_name_to_index[link_to_disable_collision])
        else:
            env.disable_collision(env.render_env.link_name_to_index[link_to_disable_collision], body=False, floor=True)

    trajectory = {
        "pcds": [],
        "flows": [],
        "3dafs": [],
        "obs_flows": [],
        "scores": [],
        "points": [],
        "directions": [],
        "masks":[],
    }
    if curate:
        trajectory["curate_mode_names"] = []

    trajectory_intermediate = {
        "pc_segs": [],
        "joint_angles": [],
        "angles": [],
    }

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
    link_pts_cnt = np.zeros(step_cnt)

    angles = []

    # Execute step_cnt steps in simulation
    for i in range(step_cnt):
        pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        link_pts_cnt[i] = len(P_world[link_ixs])
        if link_pts_cnt[i] == 0:
            p.disconnect(env.render_env.client_id)
            return None, None, None
        
        curr_angle = env.get_joint_value(target_link)

        trajectory_intermediate["joint_angles"].append(env.render_env.get_joint_angles())
        trajectory_intermediate["angles"].append(curr_angle)
        trajectory_intermediate["pc_segs"].append(pc_seg)

        curr_rel_angle = (curr_angle - init_angle) / (target_angle - init_angle)
        angles.append(curr_rel_angle)
        if i!= 0:
            trajectory["scores"].append((angles[-1] - angles[-2]) / 0.1)

        # GT Prediction
        pred_flow = gt_model(pc_obs)
        # print(pred_flow.shape)

        if curr_rel_angle > 0.98:  # End
            break

        # Whether randomize or ground truth
        if curr_rel_angle < 0.1:
            randomize = (i != step_cnt - 1) and np.random.randint(100) > int(100 * closed_gt_rate)
        else:
            randomize = (i != step_cnt - 1) and np.random.randint(100) > int(100 * open_gt_rate)

        # Create special test dataset
        if (i != step_cnt - 1) and curate_id is not None:
            if curate_id == -1:
                randomize = False
            else:
                randomize = True

        if randomize:
            now_curate = (np.random.randint(0, 2) == 0) if not curate else True
            if now_curate:
                # Curate extreme example:
                if curate_id is not None:
                    random_seed = curate_id
                else:
                    random_seed = np.random.randint(0, 3)  # 0 - same direction + min point, 2 - opposite direction + max point, 3 - opposite direction + min point
                norms = np.linalg.norm(pred_flow[link_ixs], axis=1)
                # Find the index of the maximum norm
                best_flow = pred_flow[link_ixs][np.argmax(norms)].numpy()
                # Find the index of the maximum norm
                if random_seed == 0:
                    grasp_point_idx = np.argmin(norms)
                elif random_seed == 1:
                    grasp_point_idx = np.argmax(norms)
                    best_flow = best_flow * (-1)
                else:
                    grasp_point_idx = np.argmin(norms)
                    best_flow = best_flow * (-1)
                best_point = P_world[link_ixs][grasp_point_idx]
                best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)

                if curate:
                    trajectory["curate_mode_names"].append(curate_names[obj_id][random_seed])

            else:
                # Fully random
                best_flow_ixs = [np.random.choice(len(P_world[link_ixs]))]
                best_point = P_world[link_ixs][best_flow_ixs[0]]
                best_flow = np.random.normal(size=3)
                best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)

            contact = False
            repeat_cnt = 0
            while not contact and repeat_cnt < 50:
                best_flow_ix_id, contact = env.teleport(
                    best_point[np.newaxis, :], torch.from_numpy(best_flow[np.newaxis, :]), video_writer=None, print_log=False, target_link=target_link
                )
                repeat_cnt += 1
            
            if not contact:
                p.disconnect(env.render_env.client_id)
                return None, None, None

        else:
            # best_flow_ixs, best_flows, best_points = choose_grasp_points_density(
            #     pred_flow[link_ixs], P_world[link_ixs], k=20
            # )

            # best_flow_ix_id, contact = env.teleport(
            #     best_points, best_flows, video_writer=None
            # )

            # best_flow = pred_flow[link_ixs][best_flow_ixs[best_flow_ix_id]].numpy()
            # best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]

            if len(pred_flow[link_ixs]) == 0:
                p.disconnect(env.render_env.client_id)
                return None, None, None
            
            best_flow_ixs, best_flows, best_points = choose_grasp_points(
                pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=1
            )
            best_flow = pred_flow[link_ixs][best_flow_ixs[0]].numpy()
            best_point = P_world[link_ixs][best_flow_ixs[0]]

            best_flow_ix_id = 0

            if curate:
                trajectory["curate_mode_names"].append(obj_id.split('_')[-1])


        # print(best_flow_ix_id, best_flow, best_point, contact)

        # env.attach()
        # env.pull_with_constraint(best_flow, n_steps=200, target_link=target_link, constraint=False)

        grasp_flow_field = np.zeros_like(P_world) + best_flow[np.newaxis, :]
        pcd_dist = np.power(P_world - best_point, 2).sum(axis=-1)
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
        grasp_flow_field = grasp_flow_field * coeff
        grasp_flow_field *= link_ixs[:, np.newaxis]

        trajectory["pcds"].append(P_world)
        trajectory["flows"].append(grasp_flow_field)
        trajectory["3dafs"].append(pred_flow)
        trajectory["masks"].append(link_ixs)
        trajectory["points"].append(best_point)
        trajectory["directions"].append(best_flow)


        # grasp_flow_field = np.zeros_like(P_world)
        # grasp_flow_field_segmented = np.zeros_like(P_world[link_ixs])
        # # print(grasp_flow_field[link_ixs].shape)
        # grasp_flow_field_segmented[best_flow_ixs[best_flow_ix_id]] = best_flow
        # grasp_flow_field[link_ixs] = grasp_flow_field_segmented
        animation.add_trace(
            torch.as_tensor(P_world),
            # torch.as_tensor([pcd[mask]]),
            # torch.as_tensor([flow[mask]]),
            torch.as_tensor([P_world]),
            torch.as_tensor([grasp_flow_field * 3]),
            "red",
        )


        if randomize:   # We execute the action
            env.attach()
            env.pull_with_constraint(best_flow, n_steps=100, target_link=target_link, constraint=False)

            env.reset_gripper(target_link)
            p.stepSimulation(
                env.render_env.client_id
            )  # Make sure the constraint is lifted

        else:  # We make sure that it opens the object I guess
            # env.reset_gripper(target_link)
            # p.stepSimulation(
            #     env.render_env.client_id
            # )  # Make sure the constraint is lifted
            env.set_joint_state(target_link, init_angle + (curr_rel_angle + 0.1) * (target_angle - init_angle))


        # Obtain the flow observation
        P_world_new = pma.articulate_joint(
            raw_data,
            env.render_env.get_joint_angles(),
            target_link,
            env.get_joint_value(target_link) - curr_angle,  # Articulate by only a little bit.
            P_world,
            pc_seg,
            env.render_env.link_name_to_index,
            env.render_env.T_world_base,
        )
        obs_flow = P_world_new - P_world
        trajectory["obs_flows"].append(obs_flow)



    final_angle = env.get_joint_value(target_link)
    final_rel_angle = (final_angle - init_angle) / (target_angle - init_angle)
    
    angles.append(final_rel_angle)
    trajectory["scores"].append((angles[-1] - angles[-2]) / 0.1)


    trajectory["pcds"] = np.stack(trajectory["pcds"], axis=0)
    trajectory["flows"] = np.stack(trajectory["flows"], axis=0)
    trajectory["3dafs"] = np.stack(trajectory["3dafs"], axis=0)
    trajectory["obs_flows"] = np.stack(trajectory["obs_flows"], axis=0)
    trajectory["masks"] = np.stack(trajectory["masks"], axis=0)
    trajectory["points"] = np.stack(trajectory["points"], axis=0)
    trajectory["directions"] = np.stack(trajectory["directions"], axis=0)
    trajectory["scores"] = np.array(trajectory["scores"])

    trajectory_intermediate["pc_segs"] = np.stack(trajectory_intermediate["pc_segs"], axis=0)

    # Add actions to evaluate
    evaluate_start_angle = trajectory_intermediate["angles"][-1]
    trajectory["evaluate"] = {"pcds": [], "flows":[], "3dafs":[], "scores": [], "points": [], "directions": []}
    if curate:
        trajectory["evaluate"]["curate_mode_names"] = []
    # Add history action (Rotate it to the current degree!), ground truth action
    for id, (pcd, flow, mask, pc_seg, joint_angle, angle) in enumerate(zip(trajectory["pcds"], trajectory["flows"], trajectory["masks"], trajectory_intermediate["pc_segs"], trajectory_intermediate["joint_angles"], trajectory_intermediate["angles"])):
        if id == len(trajectory["pcds"]) - 1:   # Add ground truth action to evaluate
            trajectory["evaluate"]["pcds"].append(pcd)
            trajectory["evaluate"]["flows"].append(flow)
            trajectory["evaluate"]["3dafs"].append(pred_flow.cpu().numpy())
            norms = np.linalg.norm(flow, axis=1)
            # Find the index of the maximum norm
            grasp_point_idx = np.argmax(norms)
            best_flow = flow[grasp_point_idx]
            best_point = pcd[grasp_point_idx]
            trajectory["evaluate"]["points"].append(best_point)
            trajectory["evaluate"]["directions"].append(best_flow)

            if curate:
                trajectory["evaluate"]["curate_mode_names"].append(obj_id.split('_')[-1])

        elif not curate:  # For curate samples, don't need this! (will add in these samples when adding curate samples)
            # Rotate the point cloud
            flow_end_points = pcd + flow

            # Obtain the flow observation
            new_pcd = pma.articulate_joint(
                raw_data,
                joint_angle, # env.render_env.get_joint_angles(),
                target_link,
                evaluate_start_angle - angle, # env.get_joint_value(target_link) - curr_angle,  # Articulate by only a little bit.
                pcd,
                pc_seg,
                env.render_env.link_name_to_index,
                env.render_env.T_world_base,
            )
            new_flow_end_points = pma.articulate_joint(
                raw_data,
                joint_angle, # env.render_env.get_joint_angles(),
                target_link,
                evaluate_start_angle - angle, # env.get_joint_value(target_link) - curr_angle,  # Articulate by only a little bit.
                flow_end_points,
                pc_seg,
                env.render_env.link_name_to_index,
                env.render_env.T_world_base,
            )
            new_flow = new_flow_end_points - new_pcd

            trajectory["evaluate"]["pcds"].append(new_pcd)
            trajectory["evaluate"]["flows"].append(new_flow)

            norms = np.linalg.norm(new_flow, axis=1)
            # Find the index of the maximum norm
            grasp_point_idx = np.argmax(norms)
            best_flow = new_flow[grasp_point_idx]
            best_point = new_pcd[grasp_point_idx]

            trajectory["evaluate"]["points"].append(best_point)
            trajectory["evaluate"]["directions"].append(best_flow)

            if curate:
                trajectory["evaluate"]["curate_mode_names"].append(trajectory["curate_mode_names"][id])

    # Add curate samples
    for random_seed in range(3):
        # Curate extreme example:
        # 0 - same direction + min point, 2 - opposite direction + max point, 3 - opposite direction + min point
        gt_mask = trajectory["masks"][-1]
        gt_pcd = trajectory["pcds"][-1]
        articulated_flow = normalize_trajectory(pred_flow.unsqueeze(1)).squeeze(1).numpy()
        trajectory["evaluate"]["pcds"].append(gt_pcd)

        norms = np.linalg.norm(trajectory["flows"][-1][gt_mask], axis=1)
        articulated_flow_norms = np.linalg.norm(articulated_flow, axis=1)[:, np.newaxis]
        gt_flow_mask = (articulated_flow_norms >= 0.1).squeeze()
        # print(gt_mask.shape, gt_flow_mask.shape)
        articulated_flow_norms = articulated_flow_norms[gt_flow_mask]
        # print(articulated_flow_norms)
        best_flow = pred_flow[gt_mask][np.argmax(norms)].numpy()
        # Find the index of the maximum norm
        if random_seed == 0:
            grasp_point_idx = np.argmin(norms)
            articulated_flow[gt_flow_mask] = (1 - articulated_flow_norms) * articulated_flow[gt_flow_mask] / articulated_flow_norms
        elif random_seed == 1:
            grasp_point_idx = np.argmax(norms)
            best_flow = best_flow * (-1)
            articulated_flow[gt_flow_mask] = -1 * articulated_flow[gt_flow_mask]
        else:
            grasp_point_idx = np.argmin(norms)
            best_flow = best_flow * (-1)
            articulated_flow[gt_flow_mask] = (articulated_flow_norms - 1) * articulated_flow[gt_flow_mask] / articulated_flow_norms
        best_point = gt_pcd[gt_mask][grasp_point_idx]
        best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)

        grasp_flow_field = np.zeros_like(gt_pcd) + best_flow[np.newaxis, :]
        pcd_dist = np.power(gt_pcd - best_point, 2).sum(axis=-1)
        coeff = np.exp(-1 * pcd_dist)[:, np.newaxis]
        grasp_flow_field = grasp_flow_field * coeff
        grasp_flow_field *= gt_mask[:, np.newaxis]

        trajectory["evaluate"]["flows"].append(grasp_flow_field)
        trajectory["evaluate"]["3dafs"].append(articulated_flow)
        trajectory["evaluate"]["points"].append(best_point)
        trajectory["evaluate"]["directions"].append(best_flow)

        if curate:
            trajectory["evaluate"]["curate_mode_names"].append(curate_names[obj_id][random_seed])

    if not curate:
        # Add extra random actions
        for _ in range(evaluate_random_action_cnt - 3):
            gt_mask = trajectory["masks"][-1]
            gt_pcd = trajectory["pcds"][-1]
            trajectory["evaluate"]["pcds"].append(gt_pcd)

            best_flow_ixs = [np.random.choice(len(gt_pcd[gt_mask]))]
            best_point = gt_pcd[gt_mask][best_flow_ixs[0]]
            best_flow = np.random.normal(size=3)
            best_flow /= np.maximum(np.linalg.norm(best_flow), 1e-5)

            grasp_flow_field = np.zeros_like(gt_pcd) + best_flow[np.newaxis, :]
            pcd_dist = np.power(gt_pcd - best_point, 2).sum(axis=-1)
            coeff = np.exp(-1 * pcd_dist)[:, np.newaxis]
            grasp_flow_field = grasp_flow_field * coeff
            grasp_flow_field *= gt_mask[:, np.newaxis]

            trajectory["evaluate"]["flows"].append(grasp_flow_field)
            trajectory["evaluate"]["points"].append(best_point)
            trajectory["evaluate"]["directions"].append(best_flow)

    # Evaluate the random actions and calculate scores!
    for evaluate_action_id, (pcd, flow, point, direction) in enumerate(zip(trajectory["evaluate"]["pcds"], trajectory["evaluate"]["flows"], trajectory["evaluate"]["points"], trajectory["evaluate"]["directions"])):
        # Evaluate score
        env.set_joint_state(target_link, evaluate_start_angle)
        pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]

        # Find the closest point to the selected grasp point:
        distances = np.sum((P_world - point) ** 2, axis=1)
        sorted_indices = np.argsort(distances)
        contact = False

        for idx in sorted_indices[:50]:
            grasp_point = P_world[idx]

            best_flow_ix_id, contact = env.teleport(
                grasp_point[np.newaxis, :], torch.from_numpy(direction[np.newaxis, :]), video_writer=None, print_log=False, target_link=target_link
            )
            if contact:
                break

        if not contact:
            evaluate_score = 0
        else:
            env.attach()
            env.pull_with_constraint(direction, n_steps=100, target_link=target_link, constraint=False)

            env.reset_gripper(target_link)
            p.stepSimulation(
                env.render_env.client_id
            )  # Make sure the constraint is lifted

            curr_angle = env.get_joint_value(target_link)
            evaluate_score = ((curr_angle - evaluate_start_angle) / (target_angle - init_angle)) / 0.1
        
        trajectory["evaluate"]["scores"].append(evaluate_score)


    # This is just for visualization.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=3600)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs
    # link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
    animation.add_trace(
        torch.as_tensor(P_world),
        # torch.as_tensor([pcd[mask]]),
        # torch.as_tensor([flow[mask]]),
        torch.as_tensor([P_world]),
        torch.as_tensor([np.zeros_like(P_world) * 3]),
        "red",
    )

    p.disconnect(env.render_env.client_id)
    return trajectory, animation, final_rel_angle


def generate_trajectories(obj_ids, movable_links, pm_dir, output_path, traj_per_joint, max_trials, evaluate_random_action_cnt):
    trajectories = []
    print(f"Generating for {len(obj_ids)} objects...")
    for obj_id in tqdm(obj_ids):
        if obj_id not in movable_links.keys() or len(movable_links[obj_id]) == 0:
            continue

        traj_cnt = 0
        trial_cnt = 0
        while traj_cnt < traj_per_joint and trial_cnt < max_trials:
            joint_id = np.random.choice(movable_links[obj_id])
            # for joint_id in movable_links[obj_id]:
            trajectory, _, _ = generate_trajectory(pm_dir=pm_dir, obj_id=obj_id, joint_id=joint_id, evaluate_random_action_cnt=evaluate_random_action_cnt)
            if trajectory is not None and len(trajectory["scores"]) == len(trajectory["pcds"]):
                trajectory['obj_id'] = obj_id
                trajectory['joint_id'] = joint_id
                trajectories.append(trajectory)
                traj_cnt += 1
            trial_cnt += 1
            
        # Save intermediate results
        # process_id = current_process()._identity[0] if current_process()._identity else 0
        # intermediate_path = f"{output_path}_part_{process_id}.pkl"
        with open(output_path, 'wb') as f:
            pkl.dump(trajectories, f)

        # with open(output_path.replace('.pkl', '.json'), 'w') as f:
        #     json.dump({id, obj_id}, f)

    # Final save
    with open(output_path, 'wb') as f:
        pkl.dump(trajectories, f)
    print(f"Saved trajectories to {output_path}")


def generate_trajectory_uneven(obj_id, step_cnt=-1, gui=False, normalize_pcd=False, mode="train", delta_track_path=None, delta_ckpt_path=None):
    # determine the length of the trajectory
    step_cnt = np.random.randint(1, 10) if step_cnt == -1 else step_cnt
        
    # initialize the environment
    urdf_path = os.path.join("~/datasets/unevenobject/raw", mode)
    env = ScoopEnv(rod_id = obj_id, use_GUI = gui, data_path = urdf_path, delta_track_path=delta_track_path, delta_ckpt_path=delta_ckpt_path)
    env.init_reset()
    last_action = env.prepare_scoop()

    # initialize the trajectory
    trajectory = {
        "pcds": [], # obs point cloud, here is cuboid+rod
        "flows": [], # 'delta'
        "obs_flows": [], # observed action results, P_world_new - P_world
        "scores": [], # a scalar
        "evaluate": [],
    }

    trajectory["evaluate"] = {
        "pcds": [], 
        "flows":[], 
        "scores": []
    }
    
    P_worlds = []
    
    # generate the trajectory
    for i in range(step_cnt):
        render = env.render(filter_nonobj_pts=True, n_pts=1200, normalize_pcd=normalize_pcd)
        env.update_tracker(render['obs'])
        pcd = render['P_world']
        P_worlds.append(render['P_world_org'])
        trajectory["pcds"].append(pcd)
        trajectory["evaluate"]["pcds"].append(pcd)
        
        start_step = False
        if i == step_cnt - 1:
            randomize = 1
        elif i == 0:
            randomize = 0
            start_step = True
        else:
            randomize = 0 # round(np.random.uniform(0, 1.2))

        if randomize == 0:
            action = env.choose_action_random(last_action, start_step)
        elif randomize == 1:
            action = env.ground_truth_action(last_action)
        
        flow = env.get_flow(pcd, render["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)
        trajectory["flows"].append(flow)
        trajectory["evaluate"]["flows"].append(flow)
        
        env.change_collision_with_object(turn_on=False)
        step_success = env.step(action, False)
        it_cnt = 0
        while not step_success and it_cnt < 5: # ground_truth should always succeed
            action = env.choose_action_random(action, start_step)
            step_success = env.step(action, False)
        if not step_success:
            env.close()
            return None
        env.change_collision_with_object(turn_on=True)
        action = env.elevate(action)
        env.step(action, False)
        
        render = env.render(filter_nonobj_pts=True, n_pts=1200)
        env.update_tracker(render['obs'])
        P_worlds.append(render['P_world_org'])
        
        param = 10
        obs_flow = env.get_latest_obs_flow(P_worlds[-2], normalize_pcd=normalize_pcd) * param
        trajectory["obs_flows"].append(obs_flow)
        
        reset = env.check_reset()
        if reset:
            score = -1
        else:
            score = env.get_score()
        if score == None: # no contact
            env.close()
            return None
        trajectory["scores"].append(score)
        trajectory["evaluate"]["scores"].append(score)
    
        success, _ = env.detect_success()
        
        if success and i == 0: # no history
            env.close()
            return None
        
        if reset:
            env.reset()
            last_action = env.prepare_scoop()
        else:
            last_action = env.lower(action)
            env.step(last_action, False)
            
        if success:
            break

    trajectory["pcds"] = np.stack(trajectory["pcds"], axis=0)
    trajectory["flows"] = np.stack(trajectory["flows"], axis=0)
    trajectory["obs_flows"] = np.stack(trajectory["obs_flows"], axis=0)
    trajectory["scores"] = np.array(trajectory["scores"])
    
    # get some random data
    for random_seed in range(5):
        score = None
        pcd = None
        flow = None
        while score == None:
            render = env.render(filter_nonobj_pts=True, n_pts=1200, normalize_pcd=normalize_pcd)
            pcd = render['P_world']
            
            action = env.choose_action_random(last_action)
            
            flow = env.get_flow(pcd, render["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)
                    
            env.change_collision_with_object(turn_on=False)
            step_success = env.step(action, False)
            while not step_success: # ground_truth should always succeed
                action = env.choose_action_random(action)
                env.change_collision_with_object(turn_on=False)
                step_success = env.step(action, False)
            env.change_collision_with_object(turn_on=True)
            action = env.elevate(action)
            env.step(action, False)
            
            reset = env.check_reset()
            if reset:
                score = None
                env.reset()
                last_action = env.prepare_scoop()
            else:
                score = env.get_score()
                last_action = env.lower(action)
                env.step(last_action, False)
        
        trajectory["evaluate"]["pcds"].append(pcd)
        trajectory["evaluate"]["flows"].append(flow)
        trajectory["evaluate"]["scores"].append(score)

    # uniform data
    uniform_num = 10
    for i in range(uniform_num):
        score = None
        pcd = None
        flow = None
        while score == None:
            render = env.render(filter_nonobj_pts=True, n_pts=1200, normalize_pcd=normalize_pcd)
            pcd = render['P_world']
            
            action = env.choose_action_random_uniform(last_action, i, uniform_num)
            
            flow = env.get_flow(pcd, render["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)
            
            env.change_collision_with_object(turn_on=False)
            step_success = env.step(action, False)
            while not step_success: # ground_truth should always succeed
                action = env.choose_action_random_uniform(action, i, uniform_num)
                step_success = env.step(action, False)
            env.change_collision_with_object(turn_on=True)
            action = env.elevate(action)
            env.step(action, False)
            
            reset = env.check_reset()
            if reset:
                score = None
                env.reset()
                last_action = env.prepare_scoop()
            else:
                score = env.get_score()
                last_action = env.lower(action)
                env.step(last_action, False)
                
        trajectory["evaluate"]["pcds"].append(pcd)
        trajectory["evaluate"]["flows"].append(flow)
        trajectory["evaluate"]["scores"].append(score)

    trajectory["evaluate"]["pcds"] = np.stack(trajectory["evaluate"]["pcds"], axis=0)
    trajectory["evaluate"]["flows"] = np.stack(trajectory["evaluate"]["flows"], axis=0)
    trajectory["evaluate"]["scores"] = np.array(trajectory["evaluate"]["scores"])
    
    env.close()
    return trajectory


def generate_trajectories_uneven(obj_id, output_path, traj_per_joint, max_trials, normalize_pcd, mode, delta_track_path=None, delta_ckpt_path=None):
    trajectories = []
    print(f"Generating trajectories for {obj_id}")
    
    traj_cnt = 0
    trial_cnt = 0
    while traj_cnt < traj_per_joint and trial_cnt < max_trials:
        trajectory = generate_trajectory_uneven(obj_id, normalize_pcd=normalize_pcd, mode=mode, delta_track_path=delta_track_path, delta_ckpt_path=delta_ckpt_path)
        if trajectory is not None and len(trajectory["scores"]) == len(trajectory["pcds"]):
            trajectory['obj_id'] = obj_id
            trajectories.append(trajectory)
            traj_cnt += 1
        trial_cnt += 1
        
    with open(output_path, 'wb') as f:
        pkl.dump(trajectories, f)

    print(f"Saved trajectories to {output_path}")



def main(pm_dir, output_path, train_obj_ids, test_obj_ids, movable_links, evaluate_random_action_cnt, num_processes=16, traj_per_joint_train = 200, traj_per_joint_test = 50, max_trials_train = 400, max_trials_test = 100):
    # train_obj_ids = [...]  # Replace with your train object IDs
    # test_obj_ids = [...]  # Replace with your test object IDs
    # movable_links = {...}  # Replace with your movable links data

    # Parameters for trajectory generation
    # train_output_path = '/data/failure_dataset/fullset_train_trajectory_dataset_random_action'
    # test_output_path = '/data/failure_dataset/fullset_test_trajectory_dataset_random_action'
    train_output_path = os.path.join(output_path, 'train')
    test_output_path = os.path.join(output_path, 'test')
    

    # Partition data for multiprocessing
    train_obj_chunks = [train_obj_ids[i::num_processes] for i in range(num_processes)]
    test_obj_chunks = [test_obj_ids[i::num_processes] for i in range(num_processes)]

    # Launch multiprocessing pool for training set
    with Pool(num_processes) as pool:
        pool.starmap(
            generate_trajectories,
            [(chunk, movable_links, pm_dir, f"{train_output_path}_{i}.pkl", traj_per_joint_train, max_trials_train, evaluate_random_action_cnt)
             for i, chunk in enumerate(train_obj_chunks)]
        )

    # Launch multiprocessing pool for test set
    with Pool(num_processes) as pool:
        pool.starmap(
            generate_trajectories,
            [(chunk, movable_links, pm_dir, f"{test_output_path}_{i}.pkl", traj_per_joint_test, max_trials_test, evaluate_random_action_cnt)
             for i, chunk in enumerate(test_obj_chunks)]
        )


def main_uneven(save_path, mode, traj_per_joint_train, max_trials_train, normalize_point_cloud=False, delta_track_path=None, delta_ckpt_path=None):
    if mode == 'train':
        obj_ids = [str(i) for i in range(200)]
    elif mode == 'val':
        obj_ids = [str(i) for i in range(40)]
    elif mode == 'test':
        obj_ids = [str(i) for i in range(40)]
    elif mode == 'toy':
        obj_ids = [str(i) for i in range(2,19)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Parameters for trajectory generation
    train_output_path = os.path.join(save_path, mode)
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path, exist_ok=True)

    for i, obj_id in tqdm(enumerate(obj_ids)):
        generate_trajectories_uneven(obj_id, f"{train_output_path}/{obj_id}.pkl", traj_per_joint_train, max_trials_train, normalize_point_cloud, mode, delta_track_path, delta_ckpt_path)


from tqdm import tqdm
import pickle as pkl
import json
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Generate URDF files with for Multimodal Doors.')
    # Generation parameters
    parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use for data generation.')
    parser.add_argument('--traj_per_joint_train', type=int, default=200, help='Number of trajectories to generate per joint for training set.')
    parser.add_argument('--traj_per_joint_test', type=int, default=50, help='Number of trajectories to generate per joint for test set.')
    parser.add_argument('--max_trials_train', type=int, default=400, help='Maximum trials for training set.')
    parser.add_argument('--max_trials_test', type=int, default=100, help='Maximum trials for test set.')
    # Dataset parameter
    parser.add_argument('--door', action='store_true') # Generate for door dataset only
    parser.add_argument('--dataset_path', type=str, default='/home/yishu/datasets/partnet-mobility/raw/', help='The path of the dataset.')
    parser.add_argument('--output_path', type=str, default='/data/failure_dataset_door/', help='The path to save the dataset.')
    parser.add_argument('--random_action_num', type=int, default=10, help='How many random actions to generate to consist of the evaluation action set.')
    # Uneven specific
    parser.add_argument('--uneven', action='store_true')
    return parser

if __name__ == "__main__":
    args = arg_parser().parse_args()
    full_dataset = not args.door
    pm_dir = os.path.expanduser(args.dataset_path)
    output_path = os.path.join(args.output_path, 'raw_pkls')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Articulated Objects
    if not args.uneven:
        if full_dataset:
            # Full dataset
            with open('./metadata/movable_links_fullset_000_full.json', 'r') as f:
                movable_links = json.load(f)

            with open('./metadata/umpnet_data_split_new.json', 'r') as f:
                data_split = json.load(f)
            
            train_obj_ids = [] 
            test_obj_ids = []

            for obj_id in data_split['train'].keys():
                train_obj_ids += data_split['train'][obj_id]['train']
                test_obj_ids += data_split['train'][obj_id]['test']
        else:
            # Door dataset
            with open('./metadata/multimodal_door.json', 'r') as f:
                door_split = json.load(f)

            train_obj_ids = door_split['train-train']
            test_obj_ids = door_split['test']
            movable_links = {}
            for id in train_obj_ids:
                movable_links[id] = ['_'.join(id.split('_')[1:3])]
            for id in test_obj_ids:
                movable_links[id] = ['_'.join(id.split('_')[1:3])]

        main(pm_dir, output_path, train_obj_ids, test_obj_ids, movable_links, args.random_action_num, num_processes=args.num_processes, traj_per_joint_train=args.traj_per_joint_train, traj_per_joint_test=args.traj_per_joint_test, max_trials_train=args.max_trials_train, max_trials_test=args.max_trials_test)

    else:   # Uneven objects
        delta_absolute_path = str(Path('src/have/utils/DELTA').resolve())
        if delta_absolute_path is not None and delta_absolute_path not in sys.path:
            sys.path.insert(0, delta_absolute_path)

        main_uneven(args.output_path, "toy", traj_per_joint_train=args.traj_per_joint_train, max_trials_train=args.max_trials_train, delta_track_path=delta_absolute_path, delta_ckpt_path=os.path.join(delta_absolute_path, "checkpoints/densetrack3d.pth")) # We only train on the toy rod dataset (same with training the generator)

        # Generate indices
        folder_path = os.path.join(args.output_path, "toy")

        results = {
            "data_idx": {},
            "traj_idx": {},
            "action_idx": {}
        }

        data_idx = 0 
        traj_idx = 0
        action_idx = 0

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(folder_path, file_name)
                
            try:
                with open(file_path, 'rb') as f:
                    data = pkl.load(f)
            except FileNotFoundError:
                print(f"File {file_name} not found.")
                continue

            for i, item in enumerate(data):
                for idx in range(len(item['evaluate']['scores'])):
                    results['data_idx'][f'{data_idx}'] = data[i]['obj_id']
                    results['traj_idx'][f'{traj_idx}'] = i
                    results['action_idx'][f'{action_idx}'] = idx
                    data_idx += 1
                    traj_idx += 1
                    action_idx += 1
            
            if (len(data) != 20):
                print(f'Data {file_name} has only {len(data)} trajs')

        # save the indices as a json file
        with open(os.path.join(folder_path, 'indices.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print("JSON file has been created successfully.")