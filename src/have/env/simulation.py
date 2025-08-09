# Simulation (w/ suction gripper):
# move the object according to calculated trajectory.
import os

import numpy as np
import pybullet as p
import rpad.pyg.nets.pointnet2 as pnp
import torch
from rpad.partnet_mobility_utils.data import PMObject

from failure_recovery.simulations.suction import (  # compute_flow,; run_trial_with_history,
    GTFlowModel,
    GTTrajectoryModel,
    PMSuctionSim,
    run_trial,
    run_trial_with_history_filter,
    run_trial_with_switch_models,
)

# from failure_recovery.simulations.suction_v2 import PMSuctionSim
# from failure_recovery.simulations.suction_v2 import run_trial, run_trial_with_history_filter


def trial_flow(
    obj_id="41083",
    n_steps=30,
    all_joint=True,
    available_joints=None,
    gui=False,
    website=False,
    pm_dir=os.path.expanduser("~/datasets/partnet-mobility/convex"),
    # pm_dir=os.path.expanduser("~/datasets/partnet-mobility/raw"),
):
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        sim_trajectories.append(sim_trajectory)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, sim_trajectories


# Trial with groundtruth trajectories
def trial_gt_trajectory(
    obj_id="41083",
    traj_len=10,
    n_steps=30,
    all_joint=True,
    available_joints=None,
    gui=False,
    website=False,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    movable_links = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

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

        # model = GTFlowModel(raw_data, env)
        model = GTTrajectoryModel(raw_data, env, traj_len)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            n_steps=n_steps,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
        )
        # raw_data = PMObject(os.path.join(pm_dir, obj_id))
        sim_trajectories.append(sim_trajectory)
        if result.success:
            movable_links.append(joint_name)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, movable_links, sim_trajectories

    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    # raw_data = PMObject(os.path.join(pm_dir, obj_id))

    # available_joints = raw_data.semantics.by_type("hinge") + raw_data.semantics.by_type(
    #     "slider"
    # )

    # joint = available_joints[np.random.randint(0, len(available_joints))]
    # model = GTTrajectoryModel(raw_data, env, traj_len)

    # # t0 = time.perf_counter()
    # print(f"opening {joint.name}, {joint.label}")
    # run_trial(env, raw_data, joint.name, model, n_steps=1)


def create_network(traj_len=15, ckpt_file=None):
    network = pnp.PN2Dense(
        in_channels=1, out_channels=3 * traj_len, p=pnp.PN2DenseParams()
    )
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    return network


# Trial with model predicted trajectories
def trial_with_prediction(
    obj_id="41083",
    network=None,
    n_step=1,
    gt_mask=False,
    gui=False,
    all_joint=False,
    website=False,
    sgp=True,
    available_joints=None,
    analysis=False,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)

    model = FlowSimulationInferenceModule(network, mask_input_channel=gt_mask)

    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)
        gt_model = GTFlowModel(raw_data, env) if gt_mask else None

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

        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            gt_model=gt_model,
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            sgp=sgp,
            gui=gui,
            analysis=analysis,
        )
        sim_trajectories.append(sim_trajectory)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, sim_trajectories


def trial_with_diffuser(
    obj_id="41083",
    model=None,
    n_step=30,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
    sgp=True,
    consistency_check=False,
    analysis=False,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)
    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

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

        # gt_model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial(
            env,
            raw_data,
            joint_name,
            model,
            gt_model=None,  # Don't need mask
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
            sgp=sgp,
            consistency_check=consistency_check,
            analysis=analysis,
        )
        sim_trajectories.append(sim_trajectory)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, sim_trajectories


def trial_with_diffuser_history(  # Always use sgp.... (Have already proved it's effect)
    obj_id="41083",
    model=None,
    history_model=None,
    n_step=30,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
    sgp=True,
    consistency_check=True,
    history_filter=True,
    analysis=False,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)
    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

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

        # gt_model = GTFlowModel(raw_data, env)
        # fig, result, sim_trajectory = run_trial_with_history(
        fig, result, sim_trajectory = run_trial_with_history_filter(
            env,
            raw_data,
            joint_name,
            model,
            history_model,
            gt_model=None,  # Don't need mask
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
            consistency_check=consistency_check,
            history_filter=history_filter,
            analysis=analysis,
        )
        sim_trajectories.append(sim_trajectory)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, sim_trajectories


def trial_with_switch_models(
    obj_id="41083",
    model=None,
    switch_model=None,
    history_for_models=[False, False],
    n_step=30,
    gui=False,
    all_joint=False,
    website=False,
    available_joints=None,
    return_switch_ids=False,
):
    # pm_dir = os.path.expanduser("~/datasets/partnet-mobility/raw")
    pm_dir = os.path.expanduser("~/datasets/partnet-mobility/convex")
    # env = PMSuctionSim(obj_id, pm_dir, gui=gui)
    raw_data = PMObject(os.path.join(pm_dir, obj_id))

    if available_joints is None:  # Use the passed in joint sets
        available_joints = raw_data.semantics.by_type(
            "hinge"
        ) + raw_data.semantics.by_type("slider")
        available_joints = [joint.name for joint in available_joints]

    print("available_joints:", available_joints)
    if all_joint:  # Need to traverse all the joints
        picked_joints = available_joints
    else:
        picked_joints = [available_joints[np.random.randint(0, len(available_joints))]]

    sim_trajectories = []
    results = []
    figs = {}
    for joint_name in picked_joints:
        # t0 = time.perf_counter()
        # print(f"opening {joint.name}, {joint.label}")
        print(f"opening {joint_name}")
        env = PMSuctionSim(obj_id, pm_dir, gui=gui)

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

        # gt_model = GTFlowModel(raw_data, env)
        fig, result, sim_trajectory = run_trial_with_switch_models(
            env,
            raw_data,
            joint_name,
            model,
            switch_model,
            history_for_models,
            gt_model=None,  # Don't need mask
            n_steps=n_step,
            save_name=f"{obj_id}_{joint_name}",
            website=website,
            gui=gui,
            return_switch_ids=return_switch_ids,
        )
        sim_trajectories.append(sim_trajectory)
        if result.assertion is False:
            with open(
                "/home/yishu/failure_recovery/logs/assertion_failure.txt", "a"
            ) as f:
                f.write(f"Object: {obj_id}; Joint: {joint_name}\n")
            continue
        if result.contact is False:
            continue
        figs[joint_name] = fig
        results.append(result)

    return figs, results, sim_trajectories


if __name__ == "__main__":
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
    # from failure_recovery.models.modules.dit_models import DiT

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

    # from failure_recovery.models.flow_diffuser_dit import (
    #     FlowTrajectoryDiffuserSimulationModule_DiT,
    # )

    # model = FlowTrajectoryDiffuserSimulationModule_DiT(
    #     network, inference_cfg=cfg.inference, model_cfg=cfg.model
    # ).cuda()
    # model.load_from_ckpt(ckpt_file)
    # model.eval()

    from failure_recovery.models.flow_diffuser_hispndit import (
        FlowTrajectoryDiffuserSimulationModule_HisPNDiT,
    )
    from failure_recovery.models.modules.dit_models import PN2HisDiT
    from failure_recovery.models.modules.history_encoder import HistoryEncoder

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

    # from failure_recovery.models.flow_trajectory_predictor import (
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
