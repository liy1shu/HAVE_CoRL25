# TODO: fix, and merge uneven object dataset collection code
import random
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import os
import pickle as pkl
from tqdm import tqdm
from multiprocessing import Pool
import rpad.partnet_mobility_utils.articulate as pma
from have.env.articulated.simulation import *
from have.env.articulated.suction import *
from have.env.articulated.suction import GTFlowModel, PMSuctionSim


np.random.seed(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)

full_dataset = False

closed_gt_rate = 0.2
open_gt_rate = 0.8
obj_ids = ["9127_plll", "9127_pllr", "9127_pshl", "9127_pshr"]
if full_dataset:
    pm_dir = os.path.expanduser("/home/yishu/datasets/partnet-mobility/raw/")
else:
    pm_dir = os.path.expanduser("/home/yishu/datasets/failure_history_door/raw/")
curate_names = {
    "9127_plll": ["pllr", "pshl", "pshr"],
    "9127_pllr": ["plll", "pshr", "pshl"],
    "9127_pshl": ["pshr", "plll", "pllr"],
    "9127_pshr": ["pshl", "pllr", "plll"]
}

evaluate_random_action_cnt = 10

# Action, Action result (point cloud after action), Action result (Tracking flow), Score (% of gt)
def generate_trajectory(step_cnt=-1, obj_id=-1, joint_id=0, curate=False, curate_id=None):
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


def generate_trajectories(obj_ids, movable_links, output_path, traj_per_joint, max_trials):
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
            trajectory, _, _ = generate_trajectory(obj_id=obj_id, joint_id=joint_id)
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


def main(train_obj_ids, test_obj_ids, movable_links):
    num_processes = 16
    # train_obj_ids = [...]  # Replace with your train object IDs
    # test_obj_ids = [...]  # Replace with your test object IDs
    # movable_links = {...}  # Replace with your movable links data

    # Parameters for trajectory generation
    # train_output_path = '/data/failure_dataset/fullset_train_trajectory_dataset_random_action'
    # test_output_path = '/data/failure_dataset/fullset_test_trajectory_dataset_random_action'
    train_output_path = '/data/failure_dataset_door/door_train_trajectory_dataset_random_action'
    test_output_path = '/data/failure_dataset_door/door_test_trajectory_dataset_random_action'
    traj_per_joint_train = 200
    traj_per_joint_test = 50
    max_trials_train = 400
    max_trials_test = 100

    # Partition data for multiprocessing
    train_obj_chunks = [train_obj_ids[i::num_processes] for i in range(num_processes)]
    test_obj_chunks = [test_obj_ids[i::num_processes] for i in range(num_processes)]

    # Launch multiprocessing pool for training set
    with Pool(num_processes) as pool:
        pool.starmap(
            generate_trajectories,
            [(chunk, movable_links, f"{train_output_path}_{i}.pkl", traj_per_joint_train, max_trials_train)
             for i, chunk in enumerate(train_obj_chunks)]
        )

    # Launch multiprocessing pool for test set
    with Pool(num_processes) as pool:
        pool.starmap(
            generate_trajectories,
            [(chunk, movable_links, f"{test_output_path}_{i}.pkl", traj_per_joint_test, max_trials_test)
             for i, chunk in enumerate(test_obj_chunks)]
        )


from tqdm import tqdm
import pickle as pkl
import json

if __name__ == "__main__":
    # TODO: organize everything into a config file..

    if full_dataset:
        # Full dataset
        with open('/home/yishu/failure_recovery/scripts/movable_links_fullset_000_full.json', 'r') as f:
            movable_links = json.load(f)

        with open('/home/yishu/failure_recovery/scripts/umpnet_data_split_new.json', 'r') as f:
            data_split = json.load(f)
        
        train_obj_ids = [] 
        test_obj_ids = []

        for obj_id in data_split['train'].keys():
            train_obj_ids += data_split['train'][obj_id]['train']
            test_obj_ids += data_split['train'][obj_id]['test']
    else:
        # Door dataset
        with open('/home/yishu/failure_recovery/scripts/multimodal_door.json', 'r') as f:
            door_split = json.load(f)

        train_obj_ids = door_split['train-train']
        test_obj_ids = door_split['test']
        movable_links = {}
        for id in train_obj_ids:
            movable_links[id] = ['_'.join(id.split('_')[1:3])]
        for id in test_obj_ids:
            movable_links[id] = ['_'.join(id.split('_')[1:3])]

    main(train_obj_ids, test_obj_ids, movable_links)
