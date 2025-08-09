import os
import copy
import numpy as np
import pybullet as p
import rpad.pyg.nets.pointnet2 as pnp
import imageio
import torch
import torch_geometric.data as tgd
from dataclasses import dataclass
from HAVE.simulation.ScoopEnv import ScoopEnv
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation
from flowbothd.metrics.trajectory import normalize_trajectory
from torch_geometric.data import Batch, Data

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
@dataclass
class TrialResult:
    success: bool
    assertion: bool
    metric: float
    # delta_y: float
    success_step: int
    
def trial_with_diffuser(
    obj_id="0",
    model=None,
    n_step=5,
    gui=False,
    website=False,
    analysis=False,
    model_type = 'pndit',
    oracle = '',
    scoring_model = None,
    data_path = "~/datasets/unevenobject/raw",
    normalize_pcd = False,
    record_action = False
):
    sim_trajectories = []
    results = []
    figs = {}
    sim_actions = []
    
    
    for i in range(3):
        try:
            env = ScoopEnv(rod_id = obj_id, use_GUI = gui, data_path = data_path)    
            fig, result, sim_trajectory, sim_action = run_trial(
                env,
                model,
                gt_model=None,  # Don't need mask
                n_steps=n_step,
                save_name=f"{obj_id}",
                website=website,
                gui=gui,
                analysis=analysis,
                model_type=model_type,
                oracle = oracle,
                scoring_model = scoring_model,
                normalize_pcd = normalize_pcd,
                record_action = record_action
            )
        except:
            print(f"Trial {i} failed.")
            continue
        
        if result.assertion is not False:
            break
    if result.assertion is False:
        return None, None, None, None
    figs[obj_id] = fig
    results.append(result)
    sim_trajectories.append(sim_trajectory)
    sim_actions.append(sim_action)

    return figs, results, sim_trajectories, sim_actions


def run_trial(
    env: ScoopEnv,
    model,
    gt_model=None,  # When we use mask_input_channel=True, this is the mask generator
    n_steps: int = 5,
    n_pts: int = 1200,
    save_name: str = "unknown",
    website: bool = False,
    gui: bool = False,
    analysis: bool = False,
    model_type: str = 'pndit',
    oracle = '',
    scoring_model = None,
    normalize_pcd = False,
    record_action = False
) -> TrialResult:
    
    if model_type == 'scoring':
        if scoring_model is None:
            print(f'scoring_model is None, cannot use model_type scoring')
            raise
        P_worlds = []
    
    torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    np.set_printoptions(precision=10)
    
    sim_trajectory = [0.0] + [0] * (n_steps)  # start from 0.05
    if record_action:
        sim_action = [0.0] + [0] * (n_steps) 
    
    # For website demo
    if analysis:
        visual_all_points = []
        visual_grasp_points_idx = []
        visual_grasp_points = []
        visual_flows = []

    if website:
        # Flow animation
        animation = FlowNetAnimation()

    # First, reset the environment.
    env.init_reset()
    last_action = env.prepare_scoop()

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts, normalize_pcd=normalize_pcd)
    P_world = pc_obs["P_world"]
    if model_type == 'hispndit':
        history_pcd=None
        history_flow=None

    if gt_model is None:  # GT Flow 
        if model_type == 'pndit':
            pred_trajectory = model(tuple((v for k, v in pc_obs['obs'].items())))
        elif model_type == 'hispndit':
            pred_trajectory = model(tuple((v for k, v in pc_obs['obs'].items())), history_pcd, history_flow)
        elif model_type == 'flowbot3d':
            # flowbot3d
            data = Data(pos=torch.tensor(P_world).float(), mask=torch.tensor(np.array(pc_obs["pc_seg_obj"] == 1)).float())
            batch = Batch.from_data_list([data])
            batch = batch.cuda()
            pred_trajectory = model(batch).cpu().detach().numpy()
        elif model_type == 'scoring':
            # First scoring has no history, choose an action randomly
            pred_trajectory = model(tuple((v for k, v in pc_obs['obs'].items())))
            
            env.update_tracker(pc_obs['obs'])
            P_worlds.append(pc_obs['P_world_org'])
            padded_action_pcds = torch.from_numpy(P_world).unsqueeze(0).unsqueeze(0)
    else: 
        raise
        
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    traj_len = pred_trajectory.shape[1]  # Trajectory length
    print(f"Predicting {traj_len} length trajectories.")
    pred_flow = pred_trajectory[:, 0, :]
    history_pcd=P_world

    # difference begin
    if website:
        if gui:
            # Record simulation video
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                f"./logs/simu_eval/video_assets/{save_name}.mp4",
            )
        else:
            video_file = f"./logs/simu_eval/video_assets/{save_name}.mp4"
            # cv2 output videos won't show on website
            frame_width = 640
            frame_height = 480
            
            # Camera param
            writer = imageio.get_writer(video_file, fps=5)

            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=frame_width,
                height=frame_height,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=5,
                    # yaw=180,
                    yaw=270,
                    # pitch=90,
                    pitch=-30,
                    roll=0,
                    upAxisIndex=2,
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=float(frame_width) / frame_height,
                    nearVal=0.1,
                    farVal=100.0,
                ),
            )
            image = np.array(rgbImg, dtype=np.uint8)
            image = image[:, :, :3]

            # Add the frame to the video
            writer.append_data(image)

    # The attachment point is the point with the highest flow.
    best_flow_ix, best_flow, best_point = choose_grasp_point(
        pred_flow, pc_obs["P_world_org"]
    )
    
    # For website demo
    if analysis:
        visual_all_points.append(P_world)
        visual_grasp_points_idx.append(best_flow_ix)
        visual_grasp_points.append(best_point)
        visual_flows.append(best_flow)
        
    if website:
        segmented_flow = pred_flow
        segmented_flow = np.array(
            normalize_trajectory(
                torch.from_numpy(np.expand_dims(segmented_flow, 1))
            ).squeeze()
        )
        animation.add_trace(
            torch.as_tensor(P_world),
            torch.as_tensor(np.array([P_world])),
            torch.as_tensor(np.array([segmented_flow * 3])),
            "red",
        )

    # execute
    env.change_collision_with_object(turn_on=False)
    action = last_action
    action[1] = min(max(best_point[1], env.left_most), env.right_most)
    history_flow = env.get_flow(P_world, pc_obs["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)
    if record_action:
        sim_action[1] = action[1]
    step_success = env.step(action, True, writer)
    if not step_success:
        print("not step success fail")
        p.disconnect(physicsClientId=env.physicsClientId)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                metric=0,
                success_step=0
            ),
            sim_trajectory,
            sim_action if record_action else None
        )
        
    if model_type == 'scoring':
        padded_action_flows = torch.from_numpy(env.get_flow(P_world, pc_obs["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)).unsqueeze(0).unsqueeze(0)
        
    env.change_collision_with_object(turn_on=True)
    action = env.elevate(action)
    env.step(action, True, writer)
    
    if model_type == 'scoring':
        render = env.render(filter_nonobj_pts=True, n_pts=1200)
        env.update_tracker(render['obs'])
        pcd = render['P_world']
        P_worlds.append(render['P_world_org'])
        obs_flow = env.get_latest_obs_flow(P_worlds[-2], normalize_pcd=normalize_pcd) * 10
        padded_action_results = torch.from_numpy(obs_flow).unsqueeze(0).unsqueeze(0)
        
    reset = env.check_reset()
    if reset:
        env.reset()
        last_action = env.prepare_scoop()
    else:
        last_action = action

    success, sim_trajectory[1] = env.detect_success()
    if not reset:
        last_action = env.lower(last_action)
        env.step(last_action, True, writer)
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts, normalize_pcd=normalize_pcd)
    P_world = pc_obs["P_world"]

    global_step = 1
    while not success and global_step < n_steps:
        
        # Predict the flow on the observation.
        if gt_model is None:  # GT Flow model
            if model_type == 'pndit':
                pred_trajectory = model(tuple((v for k, v in pc_obs['obs'].items())))
            elif model_type == 'hispndit':
                pred_trajectory = model(tuple((v for k, v in pc_obs['obs'].items())), history_pcd, history_flow)
            elif model_type == 'flowbot3d':
                data = Data(pos=torch.tensor(P_world).float(), mask=torch.tensor(np.array(pc_obs["pc_seg_obj"] == 1)).float())
                batch = Batch.from_data_list([data])
                batch = batch.cuda()
                pred_trajectory = model(batch).cpu().detach().numpy()
            elif model_type == 'scoring':
                # get 10 predictions
                bsz=10
                data = tgd.Data(pos=torch.from_numpy(P_world).float().cuda())
                batch = tgd.Batch.from_data_list([data] * bsz)
                pred_trajectories = model(batch)
                pred_trajectories = pred_trajectories.reshape(1200, -1, 3)
                if oracle == 'score':
                    point_ys = []
                    gt_action = env.ground_truth_action(last_action)
                    for i in range(bsz):
                        pred_flow = pred_trajectories[:, i, :]
                        best_flow_ix, best_flow, best_point = choose_grasp_point(
                            pred_flow,
                            pc_obs["P_world_org"]
                        )
                        point_ys.append(abs(best_point[1]-gt_action[1]))
                    best_idx = point_ys.index(min(point_ys))
                else:
                    if oracle == 'sampler':
                        gt_action = env.ground_truth_action(last_action)
                        gt_flow = env.get_flow(P_world, pc_obs["pc_seg_obj"], gt_action, option="torque", normalize_pcd=normalize_pcd)
                        pred_trajectories = torch.cat(
                            (pred_trajectories, torch.from_numpy(gt_flow).to(pred_trajectories.device).unsqueeze(1)), 
                            dim=1)
                        bsz+=1
                    for i in range(bsz):
                        pred_flow = pred_trajectories[:, i, :]
                        best_flow_ix, best_flow, best_point = choose_grasp_point(
                            pred_flow,
                            pc_obs["P_world_org"]
                        )
                        action_tmp = action.copy()
                        action_tmp[1] = min(max(best_point[1], env.left_most), env.right_most)
                        flow = env.get_flow(P_world, pc_obs["pc_seg_obj"], action_tmp, option="torque", normalize_pcd=normalize_pcd)
                        pred_trajectories[:, i, :] = torch.tensor(flow)
                        
                    data_list=[
                        tgd.Data(
                            pos=torch.from_numpy(P_world).float(),
                            x=pred_trajectories[:, i, :].float()
                        )
                        for i in range(bsz)
                    ]
                    action_to_evaluate = tgd.Batch.from_data_list(data_list).cuda()
                    padding_masks = torch.zeros((padded_action_pcds.shape[0]*bsz, padded_action_pcds.shape[1] + 1), dtype=torch.bool).cuda()
                    with torch.no_grad():
                        outputs, default_scores, scores = scoring_model(
                            action_to_evaluate=action_to_evaluate,
                            action_pcds=padded_action_pcds.repeat(bsz, 1, 1, 1).float().cuda(),
                            action_flows=padded_action_flows.repeat(bsz, 1, 1, 1).float().cuda(),
                            action_results=padded_action_results.repeat(bsz, 1, 1, 1).float().cuda(),
                            src_key_padding_mask=padding_masks
                        )
                    pred_scores = outputs.flatten().abs().cpu().tolist()
                    
                    best_idx = pred_scores.index(min(pred_scores))
                pred_trajectory = pred_trajectories[:, best_idx, :]
    
                env.update_tracker(pc_obs['obs'])
                P_worlds.append(pc_obs['P_world_org'])
                padded_action_pcds = torch.cat((padded_action_pcds, torch.from_numpy(P_world).unsqueeze(0).unsqueeze(0)), dim=1)
        else:
            raise
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(pred_trajectory.shape[1]):
            if global_step == n_steps:
                break
            global_step += 1
            pred_flow = pred_trajectory[:, traj_step, :]
            P_world = pc_obs["P_world"]
            history_pcd=P_world

            # Get the best direction.
            best_flow_ix, best_flow, best_point = choose_grasp_point(
                pred_flow,
                pc_obs["P_world_org"]
            )

            # For website demo
            if analysis:
                visual_all_points.append(P_world)
                visual_grasp_points_idx.append(grasp_point_id)
                visual_grasp_points.append(best_point)
                visual_flows.append(best_flow)

            # execute
            env.change_collision_with_object(turn_on=False)
            action = last_action
            action[1] = min(max(best_point[1], env.left_most), env.right_most)
            history_flow = env.get_flow(P_world, pc_obs["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)
            if record_action:
                sim_action[global_step] = action[1]
            step_success = env.step(action, True, writer)
            if not step_success:
                print("step_success fail")
                p.disconnect(physicsClientId=env.physicsClientId)
                return (
                    None,
                    TrialResult(
                        success=False,
                        assertion=False,
                        metric=0,
                        success_step=0
                    ),
                    sim_trajectory,
                    sim_action if record_action else None
                )
                
            if model_type == 'scoring':
                padded_action_flows = torch.cat((padded_action_flows, torch.from_numpy(env.get_flow(P_world, pc_obs["pc_seg_obj"], action, option="torque", normalize_pcd=normalize_pcd)).unsqueeze(0).unsqueeze(0)), dim=1)
                
            env.change_collision_with_object(turn_on=True)
            action = env.elevate(action)
            env.step(action, True, writer)
            
            if model_type == 'scoring':
                render = env.render(filter_nonobj_pts=True, n_pts=1200)
                env.update_tracker(render['obs'])
                pcd = render['P_world']
                P_worlds.append(render['P_world_org'])
                obs_flow = env.get_latest_obs_flow(P_worlds[-2], normalize_pcd=normalize_pcd) * 10
                padded_action_results = torch.cat((padded_action_results, torch.from_numpy(obs_flow).unsqueeze(0).unsqueeze(0)), dim=1)
        
            reset = env.check_reset()

            if website:
                # Add pcd to flow animation
                segmented_flow = pred_flow
                segmented_flow = np.array(
                    normalize_trajectory(
                        torch.from_numpy(np.expand_dims(segmented_flow, 1))
                    ).squeeze()
                )
                animation.add_trace(
                    torch.as_tensor(P_world),
                    torch.as_tensor(np.array([P_world])),
                    torch.as_tensor(np.array([segmented_flow * 3])),
                    "red",
                )

                # Capture frame
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=frame_width,
                    height=frame_height,
                    viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 0, 0],
                        distance=5,
                        yaw=270,
                        # yaw=90,
                        pitch=-30,
                        roll=0,
                        upAxisIndex=2,
                    ),
                    projectionMatrix=p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=float(frame_width) / frame_height,
                        nearVal=0.1,
                        farVal=100.0,
                    ),
                )
                image = np.array(rgbImg, dtype=np.uint8)
                image = image[:, :, :3]

                # Add the frame to the video
                writer.append_data(image)

            success, sim_trajectory[global_step] = env.detect_success()
    
            if success:
                for left_step in range(global_step, n_steps+1):
                    sim_trajectory[left_step] = sim_trajectory[global_step]
                break
            
            if reset:
                env.reset()
                last_action = env.prepare_scoop()
            else:
                last_action = env.lower(action)
                env.step(last_action, True, writer)
            pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts, normalize_pcd=normalize_pcd)

        if success:
            for left_step in range(global_step, n_steps+1):
                sim_trajectory[left_step] = sim_trajectory[global_step]
            break
    
    metric = env.get_score()
    if metric == None:
        print("metric is None fail!")
        p.disconnect(physicsClientId=env.physicsClientId)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                metric=0,
                success_step=0
            ),
            sim_trajectory,
            sim_action if record_action else None
        )

    if website:
        if gui:
            p.stopStateLogging(log_id)
        else:
            writer.close()

    p.disconnect(physicsClientId=env.physicsClientId)
    animation_results = None if not website else animation.animate()
    return (
        animation_results,
        TrialResult(  # Save the flow visuals
            success=success,
            assertion=True,
            metric=metric,
            success_step=global_step
        ),
        sim_trajectory
        if not analysis
        else [
            sim_trajectory,
            None,
            None,
            visual_all_points,
            visual_grasp_points_idx,
            visual_grasp_points,
            visual_flows,
        ],
        sim_action if record_action else None
    )

def choose_grasp_point(
    raw_pred_flow, 
    raw_point_cloud, 
):
    best_flow_ix = np.argmin(abs(raw_pred_flow[:, 2]))
    best_flow = raw_pred_flow[best_flow_ix]
    best_point = raw_point_cloud[best_flow_ix]
    return best_flow_ix, best_flow, best_point