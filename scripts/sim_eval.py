# TODO: merge uneven object eval code
import json
import hydra
import lightning as L
import numpy as np
import wandb
import pickle as pkl
import pandas as pd
import os
from tqdm import tqdm
import torch
import plotly.graph_objects as go
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from have.generator.models.flow_diffuser_pndit import FlowTrajectoryDiffuserInferenceModule_PNDiT
from have.generator.nets.dit_models import PN2DiT
from have.verifier.models.ha_verifier import HAVErifier
from have.env.articulated.simulation import simulation_articulated
from have.env.uneven.simulation import simulation_uneven

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Since most of us are training on 3090s+, we can use mixed precision.
torch.set_float32_matmul_precision("highest")

# Global seed for reproducibility.
L.seed_everything(42)
torch.manual_seed(42)
torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
np.set_printoptions(precision=10)


def load_models(cfg=None, generator_cfg=None, verifier_cfg=None): # TODO: add configs
    # Generator model
    network = PN2DiT(
        in_channels=generator_cfg.in_channels,
        depth=generator_cfg.depth,
        hidden_size=generator_cfg.hidden_size,
        patch_size=generator_cfg.patch_size,
        num_heads=generator_cfg.num_heads,
        n_points=cfg.dataset.n_points,
    )

    generator = FlowTrajectoryDiffuserInferenceModule_PNDiT(network, inference_cfg=cfg.inference, model_cfg=generator_cfg)
    generator = generator.to(device)

    generator.load_from_ckpt(cfg.inference.generator_ckpt)
    generator.eval()

    # Verifier model
    verifier = HAVErifier(
        d_model=verifier_cfg.d_model, 
        nhead=verifier_cfg.nhead, 
        num_layers=verifier_cfg.num_layers, 
        dim_feedforward=verifier_cfg.dim_feedforward, 
        max_len=verifier_cfg.max_len
    )

    verifier = verifier.to(device)
    state_dict = torch.load(cfg.inference.verifier_ckpt)
    state_dict = {key[7:]:value for key, value in state_dict.items()}
    verifier.load_state_dict(state_dict)  # Fullset V6.2
    verifier.scorer.test = True
    verifier.eval()
    return generator, verifier


def get_test_obj_ids(dataset_name):
    project_root = hydra.utils.get_original_cwd()  # For hydra path issue
    if "heldout" in dataset_name:
        # Full dataset simulation
        with open(os.path.join(project_root, 'metadata', 'movable_links_fullset_000_full.json'), 'r') as f:
            movable_links = json.load(f)

        with open(os.path.join(project_root, 'metadata', 'articulated_heldout.json'), 'r') as f:  # Train val unseen split
            actual_data_split = json.load(f)

        test_obj_ids = {}
        with open(os.path.join(project_root, 'metadata', 'umpnet_data_split_new.json'), 'r') as f:
            data_split = json.load(f)

            for obj_cat in data_split['train'].keys():
                # train_obj_ids += data_split['train'][obj_id]['train']
                for obj_id in (data_split['train'][obj_cat]['test'] + data_split['train'][obj_cat]['train']):

                    if obj_id in actual_data_split['train-test']:
                        # if f'val_{obj_cat}' not in test_obj_ids.keys():
                        #     test_obj_ids[f'val_{obj_cat}'] = []
                        # test_obj_ids[f'val_{obj_cat}'].append(obj_id)
                        continue
                    elif obj_id in actual_data_split['test']:
                        if f'test_{obj_cat}' not in test_obj_ids.keys():
                            test_obj_ids[f'test_{obj_cat}'] = []
                        test_obj_ids[f'test_{obj_cat}'].append(obj_id)

    elif "multimodal_door" in dataset_name:
        # Use the multimodal door dataset!
        with open(os.path.join(project_root, 'metadata', 'multimodal_door.json'), 'r') as f:
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

    elif "uneven" not in dataset_name:
        # Full dataset simulation
        with open(os.path.join(project_root, 'metadata', 'movable_links_fullset_000_full.json'), 'r') as f:
            movable_links = json.load(f)

        with open(os.path.join(project_root, 'metadata', 'umpnet_data_split_new.json'), 'r') as f:
            data_split = json.load(f)

        test_obj_ids = {}

        for obj_cat in data_split['train'].keys():
            test_obj_ids[obj_cat] = data_split['train'][obj_cat]['test']

    else:
        toy_dataset = {
            "train": [str(i) for i in range(2,19)],
            "test": [str(i) for i in range(40)] + [f"bookmark1_{i}" for i in range(20)] + [f"bookmark2_{i}" for i in range(20)] + [f"knife_{i}" for i in range(20)]
        }
        test_obj_ids = []
        for cat in ["test", "train"]:
            for id in toy_dataset[cat]:
                if "bookmark1" in id:
                    obj_type = "bookmark1"
                elif "bookmark2" in id:
                    obj_type = "bookmark2"
                elif "knife" in id:
                    obj_type = "knife"
                else:
                    obj_type = "rod" + cat
                test_obj_ids.append((id, cat, obj_type))

        movable_links = ["bookmark1", "bookmark2", "knife", "rodtrain", "rodtest", "test", "train"]

    return movable_links, test_obj_ids


def evaluate_articulated(cfg, generator, verifier, wandb_run=None):
    # Load simulation setting parameters
    multimodal_door = "door" in cfg.dataset.name
    use_tracking = cfg.tracking
    sample_cnt = cfg.sample_cnt
    oracle_sampler = cfg.oracle_sampler
    oracle_score = cfg.oracle_score
    score_filter = cfg.score_filter
    repeat_time = cfg.repeat_time

    max_step = cfg.inference.max_step
    print("Oracle Sampler:", cfg.oracle_sampler)
    print("Oracle Score:", cfg.oracle_score)
    print("Score Filter:", cfg.score_filter)

    movable_links, test_obj_ids = get_test_obj_ids(cfg.dataset.name)
    
    sample_cnts = {}
    step_cnts = {}
    step_with_grasp_cnts = {}
    steps_to_open = {}
    normalized_distance = {}
    success = {}
    mean_steps = {}
    mean_step_grasps = {}
    mean_steps_to_open = {}
    mean_steps_to_open_clean = {}
    mean_success = {}
    mean_norm_dist = {}
    fail_because_contact = {}

    columns = [
        "Object Category", "Success Rate", "Mean Norm Dist", "Mean Steps", "Mean Steps to Open"
    ]
    eval_table = wandb.Table(columns=columns)

    for obj_cat in list(test_obj_ids.keys()):
        # if obj_cat not in ["StorageFurniture"]:  # Skip
        #     continue
        if obj_cat in mean_steps.keys():
            continue

        step_cnts[obj_cat] = []
        step_with_grasp_cnts[obj_cat] = []
        steps_to_open[obj_cat] = []
        success[obj_cat] = []
        normalized_distance[obj_cat] = []
        # animations[obj_cat] = []
        fail_because_contact[obj_cat] = 0
        for obj_id in tqdm(test_obj_ids[obj_cat]):
            for joint_id in movable_links[obj_id]:
                print(obj_id, joint_id)
                for i in range(repeat_time):   # Repeat for 5 times - as in flowbothd evaluation
                    step_cnt, animation, succeed, rel_angle = simulation_articulated(
                        obj_id, 
                        sampling_model=generator, 
                        model=verifier, 
                        max_step=max_step, 
                        bsz=sample_cnt, 
                        tracking=use_tracking,
                        score_filter=score_filter, 
                        oracle_sampler=oracle_sampler, 
                        oracle_score=oracle_score, 
                        grasp_selection=False, 
                        joint_id=joint_id, 
                        camera_pos=[-4, 0, 4], 
                        multimodal_door=multimodal_door,
                        device=device
                    )
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
        sample_cnts[obj_cat] = len(success[obj_cat])
        
        # "Success Rate", "Mean Norm Dist", "Mean Steps", "Mean Steps to Open"
        eval_table.add_data(
            obj_cat,
            mean_success[obj_cat],
            mean_norm_dist[obj_cat],
            mean_steps[obj_cat],
            mean_steps_to_open[obj_cat]
        )

        # Save temporary results!
        overall_results = {
            "Success Rate": mean_success, 
            "Norm Dist": mean_norm_dist, 
            "Mean step to open": mean_steps_to_open, 
            "Mean step": mean_steps,
            "Sample Count": sample_cnts,
        }
        with open(os.path.join(wandb.run.dir, 'simulation_results.json'), 'w') as f:
            json.dump(overall_results, f, indent=4)

    print("Average Results (Across Category)")
    average_success = np.mean([mean_success[key] for key in mean_success.keys()])
    average_norm_dist = np.mean([mean_norm_dist[key] for key in mean_norm_dist.keys()])
    average_steps = np.mean([mean_steps[key] for key in mean_steps.keys()])
    average_steps_to_open = np.mean([mean_steps_to_open[key] for key in mean_steps_to_open.keys()])

    eval_table.add_data(
        "Average across Category",
        average_success,
        average_norm_dist,
        average_steps,
        average_steps_to_open
    )

    # Calculate average results across samples
    average_success = np.mean([mean_success[key] * overall_results["Sample Count"][key] for key in mean_success.keys()]) / np.sum(list(overall_results["Sample Count"].values()))
    average_norm_dist = np.mean([mean_norm_dist[key] * overall_results["Sample Count"][key] for key in mean_norm_dist.keys()]) / np.sum(list(overall_results["Sample Count"].values()))
    average_steps = np.mean([mean_steps[key] * overall_results["Sample Count"][key] for key in mean_steps.keys()]) / np.sum(list(overall_results["Sample Count"].values()))
    average_steps_to_open = np.mean([mean_steps_to_open[key] * overall_results["Sample Count"][key] for key in mean_steps_to_open.keys()]) / np.sum(list(overall_results["Sample Count"].values()))
    eval_table.add_data(
        "Average across Sample",
        average_success,
        average_norm_dist,
        average_steps,
        average_steps_to_open
    )

    wandb.log({"[SIM EVAL] Eval Table": eval_table})

    # log all of the json files
    overall_results = {
        "Success Rate": mean_success, 
        "Norm Dist": mean_norm_dist, 
        "Mean step to open": mean_steps_to_open, 
        "Mean step": mean_steps,
    }
    with open(os.path.join(wandb.run.dir, 'simulation_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)


def evaluate_uneven(cfg, generator, verifier, wandb_run=None):
    oracle = ''
    if cfg.oracle_sampler:
        oracle = 'sampler'
    elif cfg.oracle_score:
        oracle = 'score'

    # Simulation and results.
    test_obj_cats, test_obj_ids = get_test_obj_ids(cfg.dataset.name)
    metric_df = pd.DataFrame(
        np.zeros((len(test_obj_cats), 5)),
        index=test_obj_cats,
        columns=["obj_cat", "count", "success_rate", "norm_dist", "success_step"],
    )

    category_counts = {}
    sim_trajectories = {}
    sim_actions = []
    names = []

    # Create the evaluate object lists
    obj_ids = []
    obj_cats = []
    obj_types = []
    for (obj_id, obj_cat, obj_type) in test_obj_ids:
        if not os.path.exists(f"{cfg.dataset.data_dir}/raw/test/{obj_id}"):
            continue
        obj_ids.append(obj_id)
        obj_cats.append(obj_cat)
        obj_types.append(obj_type)

    for obj_id, obj_cat, obj_type in tqdm(list(zip(obj_ids, obj_cats, obj_types))):
        print(f"OBJ {obj_id} of {obj_cat}")

        for i in range(cfg.repeat_time):
            trial_figs, trial_results, sim_trajectory, sim_action = simulation_uneven(
                obj_id=obj_id,
                model=generator,
                n_step=cfg.inference.max_step,
                gui=False,
                website=False,
                model_type = 'scoring' if cfg.score_filter else 'pndit',
                oracle = oracle,
                scoring_model = verifier,
                data_path = os.path.join(cfg.dataset.data_dir, 'raw', 'test'),
                normalize_pcd = False
            )
            if trial_figs == None and trial_results == None and sim_trajectory == None:
                print("recieve Nones")
                continue

            # if obj_type not in sim_trajectories.keys():
            #     sim_trajectories[obj_type] = []
            
            # sim_trajectories[obj_type] += sim_trajectory
            sim_actions += sim_action
            names += [f"{obj_id}"]

            # Wandb table
            if obj_cat not in category_counts.keys():
                category_counts[obj_cat] = 0
            if obj_type not in category_counts.keys():
                category_counts[obj_type] = 0
            category_counts[obj_cat] += len(trial_results)
            category_counts[obj_type] += len(trial_results)
            for result in trial_results:
                print("success", result.success, "success_step", result.success_step)
                metric_df.loc[obj_cat]["success_rate"] += 1 if result.success else 0
                metric_df.loc[obj_cat]["norm_dist"] += result.metric if isinstance(result.metric, int) else result.metric.item()
                metric_df.loc[obj_cat]["success_step"] += result.success_step if result.success else 0

                metric_df.loc[obj_type]["success_rate"] += 1 if result.success else 0
                metric_df.loc[obj_type]["norm_dist"] += result.metric if isinstance(result.metric, int) else result.metric.item()
                metric_df.loc[obj_type]["success_step"] += result.success_step if result.success else 0

            if category_counts[obj_cat] == 0:
                continue
            wandb_df = metric_df.copy(deep=True)
            for cat in category_counts.keys():
                wandb_df.loc[cat]["success_step"] /=  metric_df.loc[cat]["success_rate"]
                wandb_df.loc[cat]["success_rate"] /= category_counts[cat]
                wandb_df.loc[cat]["norm_dist"] /= category_counts[cat]
                wandb_df.loc[cat]["count"] = category_counts[cat]
                wandb_df.loc[cat]["obj_cat"] = cat

            table = wandb.Table(dataframe=wandb_df.reset_index())
            wandb_run.log({f"simulation_metric_table": table})

    print(wandb_df)

    # traces = []
    # xs = list(range(31))
    # for id, (obj_type, sim_trajectory) in enumerate(sim_trajectories.items()):
    #     traces.append(
    #         go.Scatter(x=xs, y=sim_trajectory, mode="lines", name=names[id])
    #     )

    # layout = go.Layout(title="Simulation Trajectory Figure")
    # fig = go.Figure(data=traces, layout=layout)
    # wandb.log({
    #     "sim_traj_figure": wandb.Plotly(fig),
    #     "sim_traj": sim_trajectories,
    #     "sim_actions": sim_actions,
    # })

    # Save final results
    with open('simulation_results.pkl', 'wb') as f:
        pkl.dump(
            {
                "metric_df": wandb_df.to_dict(),
                # "sim_trajectories": sim_trajectories,
                "sim_actions": sim_actions,
                # "sim_traj_figure": wandb.Plotly(fig)
            },
            f,
        )



@hydra.main(config_path="../configs", config_name="eval_sim", version_base="1.3")
def main(cfg):
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
    )
    generator, verifier = load_models(cfg=cfg, generator_cfg=cfg.generator, verifier_cfg=cfg.verifier)
    if cfg.dataset.name in ["articulated_heldout", "multimodal_door", "articulated_fullset"]:
        evaluate_articulated(cfg, generator, verifier, wandb_run=run)
    else:
        evaluate_uneven(cfg, generator, verifier, wandb_run=run)
    

if __name__ == '__main__':
    main()
    

