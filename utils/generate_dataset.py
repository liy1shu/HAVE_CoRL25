import pybullet_data
import argparse
from tqdm import tqdm
import os

import pickle as pkl
import numpy as np

from multiprocessing import Pool
from HAVE.simulation.ScoopEnv import ScoopEnv

def generate_trajectory(obj_id, step_cnt=-1, gui=False, normalize_pcd=False, mode="train"):
    # determine the length of the trajectory
    step_cnt = np.random.randint(1, 10) if step_cnt == -1 else step_cnt
        
    # initialize the environment
    urdf_path = os.path.join("~/datasets/unevenobject/raw", mode)
    env = ScoopEnv(rod_id = obj_id, use_GUI = gui, data_path = urdf_path)
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


def generate_trajectories(obj_id, output_path, traj_per_joint, max_trials, normalize_pcd, mode):
    trajectories = []
    print(f"Generating trajectories for {obj_id}")
    
    traj_cnt = 0
    trial_cnt = 0
    while traj_cnt < traj_per_joint and trial_cnt < max_trials:
        trajectory = generate_trajectory(obj_id, normalize_pcd=normalize_pcd, mode=mode)
        if trajectory is not None and len(trajectory["scores"]) == len(trajectory["pcds"]):
            trajectory['obj_id'] = obj_id
            trajectories.append(trajectory)
            traj_cnt += 1
        trial_cnt += 1
        
    with open(output_path, 'wb') as f:
        pkl.dump(trajectories, f)

    print(f"Saved trajectories to {output_path}")

def main(save_path, mode, split, normalize_pcd):
    if mode == 'train':
        obj_ids = [str(i) for i in range(200)]
    elif mode == 'val':
        obj_ids = [str(i) for i in range(40)]
    elif mode == 'test':
        obj_ids = [str(i) for i in range(40)]
    elif mode == 'toy':
        obj_ids = [str(i) for i in range(2,19)]
    else:
        raise

    # Parameters for trajectory generation
    train_output_path = os.path.join(save_path, mode)
    traj_per_joint_train = 20
    max_trials_train = 200

    for i, obj_id in tqdm(enumerate(obj_ids)):
        generate_trajectories(obj_id, f"{train_output_path}/{obj_id}.pkl", traj_per_joint_train, max_trials_train, normalize_pcd, mode)

def arg_parser():
    parser = argparse.ArgumentParser(description='Generate URDF files with random parameters.')
    parser.add_argument('--save_path', type=str, default= "~/datasets/unevenobject", help='The path to save dataset.')
    parser.add_argument('--obj_path', type=str, default= "~/datasets/unevenobject/raw", help='The path to load objects.')
    parser.add_argument('--mode', type=str, default= "train", help='train, val, test')
    parser.add_argument("--normalize_pcd", action="store_true", default=False, help="Normalize pcds.")
    parser.add_argument('--split', type=str, default= "0", help='0, 1')
    return parser

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args.save_path, args.mode, args.split, args.normalize_pcd)