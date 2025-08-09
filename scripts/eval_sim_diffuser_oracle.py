# The evaluation script that runs a rollout for each object in the eval-ed dataset and calculates:
# - success : 90% open
# - distance to open

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
import json
import os

import hydra
import lightning as L
import numpy as np
import omegaconf
import pandas as pd
import plotly.graph_objects as go
import rpad.pyg.nets.pointnet2 as pnp
import torch
import tqdm
import wandb
from rpad.visualize_3d import html

from flowbothd.models.flow_diffuser_dgdit import (
    FlowTrajectoryDiffuserSimulationModule_DGDiT,
)
from flowbothd.models.flow_diffuser_dit import (
    FlowTrajectoryDiffuserSimulationModule_DiT,
)
from flowbothd.models.flow_diffuser_pndit import (
    FlowTrajectoryDiffuserSimulationModule_PNDiT,
)
from flowbothd.models.flow_trajectory_diffuser import (
    FlowTrajectoryDiffuserSimulationModule_PN2,
)
from flowbothd.models.modules.dit_models import DGDiT, DiT, PN2DiT
from HAVE.simulation.simulation import trial_with_diffuser
from flowbothd.utils.script_utils import PROJECT_ROOT, match_fn

from scripts.scoring_module_flash_qkv_accelerator_v2 import TransformerModel

print(PROJECT_ROOT)


def load_obj_id_to_category(toy_dataset=None):
    id_to_cat = {}
    if toy_dataset is None:
        # Extract existing classes.
        with open(f"{PROJECT_ROOT}/scripts/umpnet_data_split_new.json", "r") as f:
            data = json.load(f)

        for _, category_dict in data.items():
            for category, split_dict in category_dict.items():
                for train_or_test, id_list in split_dict.items():
                    for id in id_list:
                        id_to_cat[id] = f"{category}_{train_or_test}"

    else:
        for split in ["test"]: # "train-train", "train-test", "test"
            for id in toy_dataset[split]:
                id_to_cat[id] = split
    return id_to_cat


def load_obj_and_link(id_to_cat):
    with open(f"{PROJECT_ROOT}/scripts/movable_links_fullset_000.json", "r") as f:
        object_link_json = json.load(f)
    for id in id_to_cat.keys():
        if id not in object_link_json.keys():
            object_link_json[id] = []
    return object_link_json


inference_module_class = {
    "diffuser_pn++": FlowTrajectoryDiffuserSimulationModule_PN2,
    "diffuser_dgdit": FlowTrajectoryDiffuserSimulationModule_DGDiT,
    "diffuser_dit": FlowTrajectoryDiffuserSimulationModule_DiT,
    "diffuser_pndit": FlowTrajectoryDiffuserSimulationModule_PNDiT,
}


@hydra.main(config_path="../configs", config_name="eval_sim_uneven_object_scoring", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    L.seed_everything(42)

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################
   
    
    if cfg.dataset.dataset_type == "full-dataset":
        # Full dataset
        toy_dataset = None
    elif cfg.dataset.dataset_type == "uneven-object":
        toy_dataset = {
            "id": "uneven-object",
            "train-train": [str(i) for i in range(200)],
            "train-test": [str(i) for i in range(40)],
            "test": [str(i) for i in range(40)] + [f"bookmark1_{i}" for i in range(20)] + [f"bookmark2_{i}" for i in range(20)] + [f"knife_{i}" for i in range(20)],
            "test-toy": [str(i) for i in range(2,19)]
        }
    else:
        # Door dataset
        toy_dataset = {
            "id": "door-full-new-noslide",
            "train-train": [
                "8877",
                "8893",
                "8897",
                "8903",
                "8919",
                "8930",
                "8961",
                "8997",
                "9016",
                # "9032",   # has slide
                "9035",
                "9041",
                "9065",
                "9070",
                "9107",
                "9117",
                "9127",
                "9128",
                "9148",
                "9164",
                "9168",
                "9277",
                "9280",
                "9281",
                "9288",
                "9386",
                "9388",
                "9410",
            ],
            "train-test": ["8867", "8983", "8994", "9003", "9263", "9393"],
            "test": ["8867", "8983", "8994", "9003", "9263", "9393"],
        }

    id_to_cat = load_obj_id_to_category(toy_dataset)
    # object_to_link = load_obj_and_link(id_to_cat)
    ######################################################################
    # Set up logging in WandB.
    # This is a different job type (eval), but we want it all grouped
    # together. Notice that we use our own logging here (not lightning).
    ######################################################################

    # Create a run.
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=cfg.wandb.group,
        # mode = 'offline'
    )

    # Log the code.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    trajectory_len = cfg.inference.trajectory_len
    if "diffuser" in cfg.model.name:
        if "pn++" in cfg.model.name:
            in_channels = 3 * cfg.inference.trajectory_len + cfg.model.time_embed_dim
        else:
            in_channels = (
                3 * cfg.inference.trajectory_len
            )  # Will add 3 as input channel in diffuser
    else:
        in_channels = 1 if cfg.inference.mask_input_channel else 0

    if "pn++" in cfg.model.name:
        network = pnp.PN2Dense(
            in_channels=in_channels,
            out_channels=3 * trajectory_len,
            p=pnp.PN2DenseParams(),
        ).cuda()
    elif "dgdit" in cfg.model.name:
        network = DGDiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=cfg.dataset.n_points,
        ).cuda()
    elif "pndit" in cfg.model.name:
        network = PN2DiT(
            in_channels=in_channels,
            depth=5,
            hidden_size=128,
            patch_size=1,
            num_heads=4,
            n_points=cfg.dataset.n_points,
        ).cuda()
    elif "dit" in cfg.model.name:
        network = DiT(
            in_channels=in_channels + 3,
            depth=5,
            hidden_size=128,
            num_heads=4,
            learn_sigma=True,
        ).cuda()

    # Load the network weights.
    ckpt_file = 'your ckpt file path'
    model = inference_module_class[cfg.model.name](
        network, inference_cfg=cfg.inference, model_cfg=cfg.model
    ).cuda()
    model.load_from_ckpt(ckpt_file)
    model.eval()
    
    scoring_model = TransformerModel(d_model=128, nhead=4, num_layers=4, dim_feedforward=256, max_len=200)
    scoring_model = scoring_model.cuda()
    scoring_ckpt_file = 'path to the scoring model'
    state_dict = torch.load(scoring_ckpt_file, map_location="cuda:0")
    new_state_dict = {key[7:]: value for key, value in state_dict.items()}
    scoring_model.load_state_dict(new_state_dict)  # Load the state_dict into the scoring_model
    scoring_model.scorer.test = True
    scoring_model.eval()

    # Simulation and results.
    print("Simulating")
    if cfg.website:
        # Visualization html
        os.makedirs("./logs/simu_eval/video_assets/")
        doc = html.PlotlyWebsiteBuilder("Simulation Visualizations")
    obj_cats = list(set(id_to_cat.values()))
    metric_df = pd.DataFrame(
        np.zeros((len(set(id_to_cat.values())), 5)),
        index=obj_cats,
        columns=["obj_cat", "count", "success_rate", "norm_dist", "success_step"],
    )
    category_counts = {}
    sim_trajectories = []
    sim_actions = []
    names = []

    # Create the evaluate object lists
    repeat_time = 3
    obj_ids = []
    for obj_id, obj_cat in tqdm.tqdm(list(id_to_cat.items())):
        if "test" not in obj_cat:
            continue
        if obj_cat == "train-test":
            dir_name = 'val'
        elif obj_cat == "test":
            dir_name = 'test'
        elif obj_cat == "test-toy":
            dir_name = 'toy'
        if not os.path.exists(f"{cfg.dataset.data_dir}/raw/{dir_name}/{obj_id}"):
            continue
        obj_ids.append(obj_id)
    obj_ids = obj_ids * repeat_time

    import random

    random.shuffle(obj_ids)

    for obj_id in tqdm.tqdm(obj_ids):
        obj_cat = id_to_cat[obj_id]
        if obj_cat == "train-test":
            dir_name = 'val'
        elif obj_cat == "test":
            dir_name = 'test'
        elif obj_cat == "test-toy":
            dir_name = 'toy'
        print(f"OBJ {obj_id} of {obj_cat}")
        trial_figs, trial_results, sim_trajectory, sim_action = trial_with_diffuser(
            obj_id=obj_id,
            model=model,
            n_step=10,
            gui=False,
            website=cfg.website,
            model_type = 'scoring',
            oracle = 'score',
            scoring_model = scoring_model,
            data_path = f"~/datasets/unevenobject/raw/{dir_name}",
            normalize_pcd = False
        )
        if trial_figs == None and trial_results == None and sim_trajectory == None:
            continue
        sim_trajectories += sim_trajectory
        sim_actions += sim_action
        names += [f"{obj_id}"]

        # Wandb table
        if obj_cat not in category_counts.keys():
            category_counts[obj_cat] = 0
        category_counts[obj_cat] += len(trial_results)
        for result in trial_results:
            print("success", result.success, "success_step", result.success_step)
            metric_df.loc[obj_cat]["success_rate"] += 1 if result.success else 0
            metric_df.loc[obj_cat]["norm_dist"] += result.metric.item()
            metric_df.loc[obj_cat]["success_step"] += result.success_step if result.success else 0
        if cfg.website: # no joints
            # Website visualization
            for id, (obj_id_, fig) in enumerate(trial_figs.items()):
                tag = f"{obj_id}"
                if fig is not None:
                    doc.add_plot(obj_cat, tag, fig)
                doc.add_video(
                    obj_cat,
                    f"{tag}",
                    f"http://128.2.178.238:{cfg.website_port}/video_assets/{tag}.mp4",
                )
            doc.write_site("./logs/simu_eval")
        
        if category_counts[obj_cat] == 0:
            continue
        wandb_df = metric_df.copy(deep=True)
        for obj_cat in category_counts.keys():
            wandb_df.loc[obj_cat]["success_step"] /=  metric_df.loc[obj_cat]["success_rate"]
            wandb_df.loc[obj_cat]["success_rate"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["norm_dist"] /= category_counts[obj_cat]
            wandb_df.loc[obj_cat]["count"] = category_counts[obj_cat]
            wandb_df.loc[obj_cat]["obj_cat"] = 0 if "train" in obj_cat else 1

        table = wandb.Table(dataframe=wandb_df.reset_index())
        run.log({f"simulation_metric_table": table})

    print(wandb_df)

    traces = []
    xs = list(range(31))
    for id, sim_trajectory in enumerate(sim_trajectories):
        traces.append(
            go.Scatter(x=xs, y=sim_trajectory, mode="lines", name=names[id])
        )

    layout = go.Layout(title="Simulation Trajectory Figure")
    fig = go.Figure(data=traces, layout=layout)
    wandb.log({
        "sim_traj_figure": wandb.Plotly(fig),
        "sim_traj": sim_trajectories,
        "sim_actions": sim_actions,
        })


if __name__ == "__main__":
    main()
