from typing import Any, Dict

import lightning as L
import plotly.express as px
import plotly.graph_objects as go
import rpad.visualize_3d.plots as v3p
import torch
import torch_geometric.data as tgd
import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from plotly.subplots import make_subplots
from torch import optim

# from flowbothd.models.modules.dit_models import DiT
from failure_recovery.metrics.trajectory import (
    artflownet_loss,
    flow_metrics,
    normalize_trajectory,
)
from failure_recovery.models.dit_utils import create_diffusion


# Flow predictor with PN++ + DiT
class FlowTrajectoryDiffusionModule_PNDiT(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg) -> None:
        super().__init__()

        # Training params
        self.batch_size = training_cfg.batch_size
        self.lr = training_cfg.lr
        self.mode = training_cfg.mode
        self.traj_len = training_cfg.trajectory_len
        self.epochs = training_cfg.epochs
        self.train_sample_number = training_cfg.train_sample_number

        # Diffuser training param
        self.wta = training_cfg.wta
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # Diffuser params
        self.sample_size = 1200
        self.wta_trial_times = training_cfg.wta_trial_times

        self.backbone = network
        self.num_train_timesteps = model_cfg.num_train_timesteps
        self.diffusion = create_diffusion(
            timestep_respacing=None, diffusion_steps=self.num_train_timesteps
        )

        self.cosine_distribution_cache = {"x": [], "y": [], "colors": []}

    def set_dataloader_names(self, new_names):
        self.dataloader_names = new_names

    def forward(self, batch: tgd.Batch, mode):
        x = (
            batch.delta.squeeze(2)
            .reshape(-1, 30, 40, 3 * self.traj_len)
            .permute(0, 3, 1, 2)
            .float()
            .cuda()
        )
        pos = batch.pos.reshape(-1, self.sample_size, 3 * self.traj_len).float().cuda()

        model_kwargs = dict(pos=pos, context=batch.cuda())
        loss_dict = self.diffusion.training_losses(
            self.backbone, x, batch.timesteps, model_kwargs
        )
        loss = loss_dict["loss"].mean()

        self.log_dict(
            {
                f"{mode}/loss": loss,
                # The other metrics will be tested in validation
                # f"{mode}/rmse": rmse,
                # f"{mode}/cosine_similarity": cos_dist,
                # f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return None, loss

    # @torch.inference_mode()
    def predict(self, batch: tgd.Batch, mode):
        # torch.eval()
        bs = batch.delta.shape[0] // self.sample_size
        z = torch.randn(bs, 3 * self.traj_len, 30, 40, device=self.device)  # .float()

        pos = batch.pos.reshape(bs, self.sample_size, 3 * self.traj_len).float().cuda()
        model_kwargs = dict(pos=pos, context=batch)

        samples, results = self.diffusion.p_sample_loop(
            self.backbone,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )

        f_pred = (
            torch.flatten(samples, start_dim=2, end_dim=3)
            .permute(0, 2, 1)
            .reshape(-1, 3 * self.traj_len)
            .unsqueeze(1)
        )
        f_pred = normalize_trajectory(f_pred)

        # Compute the loss.
        # mask = batch.mask == 1
        # mask = mask.reshape(-1, self.sample_size).to("cuda")

        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point

        f_target = f_target  # .float()
        f_target = normalize_trajectory(f_target)

        # print(f_pred[f_ix], batch.delta[f_ix])
        loss = artflownet_loss(f_pred, f_target, n_nodes)

        if torch.sum(f_ix) == 0:  # No point
            return f_pred, loss

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(f_pred[f_ix], f_target[f_ix])

        self.log_dict(
            {
                f"{mode}/flow_loss": loss,
                f"{mode}/rmse": rmse,
                f"{mode}/cosine_similarity": cos_dist,
                f"{mode}/mag_error": mag_error,
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return f_pred, loss

    def predict_wta(self, orig_batch: tgd.Batch, mode):
        bs = orig_batch.delta.shape[0] // self.sample_size
        assert bs == 1, "Only support bsz = 1 for winner take all evaluation"
        bs = self.wta_trial_times
        data_list = orig_batch.to_data_list() * bs
        batch = tgd.Batch.from_data_list(data_list)

        z = torch.randn(bs, 3 * self.traj_len, 30, 40, device=self.device)  # .float()

        pos = batch.pos.reshape(bs, self.sample_size, 3 * self.traj_len).float().cuda()
        model_kwargs = dict(pos=pos, context=batch)

        samples, results = self.diffusion.p_sample_loop(
            self.backbone,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )

        f_pred = (
            torch.flatten(samples, start_dim=2, end_dim=3)
            .permute(0, 2, 1)
            .reshape(-1, 3 * self.traj_len)
            .unsqueeze(1)
        )
        f_pred = normalize_trajectory(f_pred)

        # Compute the loss.
        mask = batch.mask == 1
        mask = mask.reshape(-1, self.sample_size).to("cuda")

        n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
        f_ix = batch.mask.bool()
        if self.mode == "delta":
            f_target = batch.delta
        elif self.mode == "point":
            f_target = batch.point

        f_target = f_target  # .float()
        f_target = normalize_trajectory(f_target)

        # print(f_pred[f_ix], batch.delta[f_ix])
        loss = artflownet_loss(f_pred, f_target, n_nodes, reduce=False)
        flow_loss = loss.reshape(bs, -1).mean(-1)
        chosen_id = torch.min(flow_loss, 0)[1]  # index

        if torch.sum(f_ix) == 0:  # No point
            return (
                f_pred.reshape(bs, self.sample_size, self.traj_len, 3)[chosen_id],
                loss[chosen_id],
                [],
            )

        # Compute some metrics on flow-only regions.
        rmse, cos_dist, mag_error = flow_metrics(
            f_pred[f_ix], f_target[f_ix], reduce=False
        )

        # Aggregate the results
        # Choose the one with smallest flow loss
        rmse = rmse.reshape(bs, -1).mean(-1)
        cos_dist = cos_dist.reshape(bs, -1).mean(-1)
        mag_error = mag_error.reshape(bs, -1).mean(-1)

        pos_cosine = torch.sum((cos_dist - 0.7) > 0) / bs
        neg_cosine = torch.sum((cos_dist + 0.7) < 0) / bs
        multimodal = 1 if (pos_cosine != 0 and neg_cosine != 0) else 0

        self.log_dict(
            {
                f"{mode}_wta/flow_loss": flow_loss[chosen_id].item(),
                f"{mode}_wta/rmse": rmse[chosen_id].item(),
                f"{mode}_wta/cosine_similarity": cos_dist[chosen_id].item(),
                f"{mode}_wta/mag_error": mag_error[chosen_id].item(),
                f"{mode}_wta/multimodal": multimodal,
                f"{mode}_wta/pos@0.7": pos_cosine.item(),
                f"{mode}_wta/neg@0.7": neg_cosine.item(),
            },
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return (
            f_pred.reshape(bs, self.sample_size, self.traj_len, 3)[chosen_id],
            loss[chosen_id],
            cos_dist.tolist(),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(
                (self.train_sample_number // self.batch_size) * self.epochs
            ),
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tgd.Batch, batch_id):  # type: ignore
        self.train()
        bs = batch.delta.shape[0] // self.sample_size

        # batch.delta = normalize_trajectory(batch.delta)
        batch.timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (bs,),
            device=self.device,
        ).long()

        _, loss = self(batch, "train")
        return loss

    def validation_step(self, batch: tgd.Batch, batch_id, dataloader_idx=0):  # type: ignore
        self.eval()

        # Clean cache for a new eval dataloader
        if batch_id == 0:
            self.cosine_distribution_cache["x"] = []
            self.cosine_distribution_cache["y"] = []
            self.cosine_distribution_cache["colors"] = []

        # dataloader_names = ["val", "train", "unseen"]
        name = self.dataloader_names[dataloader_idx]
        with torch.no_grad():
            f_pred, loss = self.predict(batch, name)
            # print("predict:", f_pred.shape)
            if self.wta:
                f_pred, loss, cosines = self.predict_wta(batch, name)
                self.cosine_distribution_cache["x"] += [batch_id] * len(cosines)
                self.cosine_distribution_cache["y"] += cosines
                self.cosine_distribution_cache["colors"] += [
                    "blue" if batch_id % 2 == 0 else "red"
                ] * len(cosines)
        # breakpoint()
        return {
            "preds": f_pred,
            "loss": loss,
            "cosine_cache": self.cosine_distribution_cache,
        }

    @staticmethod
    def make_plots(preds, batch: tgd.Batch, cosine_cache=None) -> Dict[str, go.Figure]:
        # 1) Make the flow visualization plots
        obj_id = batch.id
        pos = (
            batch.point[:, -2, :].numpy() if batch.point.shape[1] >= 2 else batch.pos
        )  # The last step's beinning pos
        mask = batch.mask.numpy()
        f_target = batch.delta[:, -1, :]
        f_pred = preds.reshape(preds.shape[0], -1, 3)[:, -1, :]

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "input data",
                "target flow",
                "pred flow",
            ),
            vertical_spacing=0.05,
        )

        # Parent/child plot.
        labelmap = {0: "unselected", 1: "part"}
        labels = torch.zeros(len(pos)).int()
        labels[mask == 1.0] = 1
        fig.add_traces(v3p._segmentation_traces(pos, labels, labelmap, "scene1"))

        fig.update_layout(
            scene1=v3p._3d_scene(pos),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=1.0, y=0.75),
        )

        # normalize the flow for visualization.
        n_f_gt = (f_target / f_target.norm(dim=1).max()).numpy()
        n_f_pred = (f_pred / f_target.norm(dim=1).max()).numpy()

        # GT flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene2", name="pts"), row=2, col=1)
        f_gt_traces = v3p._flow_traces(
            pos, n_f_gt, scene="scene2", name="f_gt", legendgroup="1"
        )
        fig.add_traces(f_gt_traces, rows=2, cols=1)
        fig.update_layout(scene2=v3p._3d_scene(pos))

        # Predicted flow.
        fig.add_trace(v3p.pointcloud(pos, 1, scene="scene3", name="pts"), row=2, col=2)
        f_pred_traces = v3p._flow_traces(
            pos, n_f_pred, scene="scene3", name="f_pred", legendgroup="2"
        )
        fig.add_traces(f_pred_traces, rows=2, cols=2)
        fig.update_layout(scene3=v3p._3d_scene(pos))

        fig.update_layout(title=f"Object {obj_id}")

        # 2) Make the cosine distribution plots
        # table = wandb.Table(data=data, columns=["class_x", "class_y"])
        # wandb.log({"my_custom_id": wandb.plot.scatter(table, "class_x", "class_y")})
        cos_fig = None
        if (
            cosine_cache is not None and len(cosine_cache["x"]) != 0
        ):  # Does wta, and needs plots
            # The following Matplotlib code won't work because some matplotlib version issue (3.4.3 would work, but the version is old)
            # cos_fig = plt.figure()
            # ax = cos_fig.add_axes([0.1, 0.1, 0.8, 0.8])
            # plt.ylim((-1, 1))
            # ax.axhline(y=0)
            # ax.axhline(y=0.7)
            # ax.axhline(y=-0.7)
            # plt.scatter(cosine_cache["x"], cosine_cache["y"], s=5, c=cosine_cache["colors"])
            cos_fig = px.scatter(
                x=cosine_cache["x"], y=cosine_cache["y"], color=cosine_cache["colors"]
            )
            cos_fig.update_layout(yaxis_range=[-1, 1])

        return {"diffuser_plot": fig, "cosine_distribution_plot": cos_fig}


class FlowTrajectoryDiffuserInferenceModule_PNDiT(L.LightningModule):
    def __init__(self, network, inference_cfg, model_cfg) -> None:
        super().__init__()
        # Inference params
        self.batch_size = inference_cfg.batch_size
        self.traj_len = inference_cfg.trajectory_len

        # Diffuser params
        self.sample_size = 1200
        self.backbone = network
        self.num_inference_timesteps = model_cfg.num_train_timesteps
        self.diffusion = create_diffusion(
            timestep_respacing=None, diffusion_steps=self.num_inference_timesteps
        )

    def load_from_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.load_state_dict(ckpt["state_dict"])

    def forward(self, data) -> torch.Tensor:  # type: ignore
        print(
            "Don't call this, it's not implemented. You should call predict_step or predict_wta"
        )
        return None

    def predict(self, P_world, bsz=1, cond_fn=None, classifier_inputs=None, return_intermediate=False, start_noise=None) -> torch.Tensor:  # From pure point cloud
        data = tgd.Data(
            pos=torch.from_numpy(P_world).float().cuda(),
            # mask=torch.ones(P_world.shape[0]).float(),
        )
        batch = tgd.Batch.from_data_list([data] * bsz)
        # batch = batch.to(self.device)
        # batch.x = batch.mask.reshape(len(batch.mask), 1)
        self.eval()
        with torch.no_grad():
            # trajectory = self.model.faster_predict_step(batch, 0)
            trajectory = self.predict_step(batch, 0, cond_fn=cond_fn, classifier_inputs=classifier_inputs, return_intermediate=return_intermediate, start_noise=start_noise)
        if return_intermediate:
            trajectory, intermediates = trajectory
        # print("Trajectory prediction shape:", trajectory.shape)
            return trajectory.cpu(), intermediates
        else:
            return trajectory.cpu()


    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, cond_fn: Any = None, classifier_inputs: Any = None, return_intermediate=False, start_noise=None) -> torch.Tensor:  # type: ignore
        # torch.eval()
        self.eval()
        bs = batch.pos.shape[0] // self.sample_size
        if start_noise is None:
            z = torch.randn(bs, 3 * self.traj_len, 30, 40, device=self.device)  # .float()
        else:
            z = start_noise.to(self.device)

        pos = batch.pos.reshape(bs, self.sample_size, 3 * self.traj_len).float().cuda()
        if classifier_inputs is None:
            model_kwargs = dict(pos=pos, context=batch.cuda())
        else:
            model_kwargs = dict(pos=pos, context=batch.cuda(), **classifier_inputs)

        samples, results = self.diffusion.p_sample_loop(
            self.backbone,
            z.shape,
            z,
            clip_denoised=False,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )

        f_pred = (
            torch.flatten(samples, start_dim=2, end_dim=3)
            .permute(0, 2, 1)
            .reshape(-1, 3 * self.traj_len)
            .unsqueeze(1)
        )
        f_pred = normalize_trajectory(f_pred)
        if return_intermediate:
            return f_pred, results
        else:
            return f_pred

    # For winner takes it all evaluation
    @torch.inference_mode()
    def predict_wta(self, dataloader, mode="delta", trial_times=50):
        all_rmse = 0
        all_cos_dist = 0
        all_mag_error = 0
        all_flow_loss = 0
        all_multimodal = 0
        all_pos_cosine = 0
        all_neg_cosine = 0
        valid_sample_cnt = 0

        for id, orig_sample in tqdm.tqdm(enumerate(dataloader)):
            bs = orig_sample.delta.shape[0] // self.sample_size
            assert bs == 1, f"batch size should be 1, now is {bs}"

            # batch every sample into bsz of trial_times
            bs = trial_times
            data_list = orig_sample.to_data_list() * bs
            batch = tgd.Batch.from_data_list(data_list)

            z = torch.randn(
                bs, 3 * self.traj_len, 30, 40, device=self.device
            )  # .float()

            pos = (
                batch.pos.reshape(bs, self.sample_size, 3 * self.traj_len)
                .float()
                .to(self.device)
            )
            model_kwargs = dict(pos=pos, context=batch)
            # breakpoint()
            samples, results = self.diffusion.p_sample_loop(
                self.backbone,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=self.device,
            )

            f_pred = (
                torch.flatten(samples, start_dim=2, end_dim=3)
                .permute(0, 2, 1)
                .reshape(-1, 3 * self.traj_len)
                .unsqueeze(1)
            )
            f_pred = normalize_trajectory(f_pred)

            # Compute the loss.
            mask = batch.mask == 1
            mask = mask.reshape(-1, self.sample_size).to("cuda")

            n_nodes = torch.as_tensor([d.num_nodes for d in batch.to_data_list()]).to(self.device)  # type: ignore
            f_ix = batch.mask.bool().to(self.device)
            if torch.sum(f_ix) == 0:
                continue

            valid_sample_cnt += 1
            if mode == "delta":
                f_target = batch.delta.to(self.device)
            elif mode == "point":
                f_target = batch.point.to(self.device)

            f_target = f_target  # .float()
            f_target = normalize_trajectory(f_target)

            # print(f_pred[f_ix], batch.delta[f_ix])
            # print(f_pred.device, f_target.device)
            flow_loss = artflownet_loss(f_pred, f_target, n_nodes, reduce=False)

            # Compute some metrics on flow-only regions.
            rmse, cos_dist, mag_error = flow_metrics(
                f_pred[f_ix], f_target[f_ix], reduce=False
            )

            # Aggregate the results
            # Choose the one with smallest flow loss
            flow_loss = flow_loss.reshape(bs, -1).mean(-1)
            rmse = rmse.reshape(bs, -1).mean(-1)
            cos_dist = cos_dist.reshape(bs, -1).mean(-1)
            mag_error = mag_error.reshape(bs, -1).mean(-1)

            chosen_id = torch.min(flow_loss, 0)[1]  # index
            pos_cosine = torch.sum((cos_dist - 0.7) > 0) / bs
            neg_cosine = torch.sum((cos_dist + 0.7) < 0) / bs
            multimodal = 1 if (pos_cosine != 0 and neg_cosine != 0) else 0

            print(
                multimodal,
                rmse[chosen_id],
                cos_dist[chosen_id],
                mag_error[chosen_id],
                flow_loss[chosen_id],
            )

            all_multimodal += multimodal
            all_pos_cosine += pos_cosine.item()
            all_neg_cosine += neg_cosine.item()
            all_rmse += rmse[chosen_id].item()
            all_cos_dist += cos_dist[chosen_id].item()
            all_mag_error += mag_error[chosen_id].item()
            all_flow_loss += flow_loss[chosen_id].item()

        metric_dict = {
            f"flow_loss": all_flow_loss / valid_sample_cnt,  # / len(dataloader),
            f"rmse": all_rmse / valid_sample_cnt,  # / len(dataloader),
            f"cosine_similarity": all_cos_dist / valid_sample_cnt,  # / len(dataloader),
            f"mag_error": all_mag_error / valid_sample_cnt,  # / len(dataloader),
            f"multimodal": all_multimodal / valid_sample_cnt,  # / len(dataloader),
            f"pos@0.7": all_pos_cosine / valid_sample_cnt,  # / len(dataloader),
            f"neg@0.7": all_neg_cosine / valid_sample_cnt,  # / len(dataloader),
        }

        self.log_dict(
            metric_dict,
            add_dataloader_idx=False,
            batch_size=len(batch),
        )
        return metric_dict, cos_dist.tolist()  # dataloader * trial_times



class FlowTrajectoryDiffuserSimulationModule_PNDiT(L.LightningModule):
    def __init__(self, network, inference_cfg, model_cfg) -> None:
        super().__init__()
        self.model = FlowTrajectoryDiffuserInferenceModule_PNDiT(
            network, inference_cfg, model_cfg
        )

    def load_from_ckpt(self, ckpt_file):
        self.model.load_from_ckpt(ckpt_file)

    def forward(self, data) -> torch.Tensor:  # type: ignore
        # Maybe add the mask as an input to the network.
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = data

        data = tgd.Data(
            pos=torch.from_numpy(P_world).float().cuda(),
            # mask=torch.ones(P_world.shape[0]).float(),
        )
        batch = tgd.Batch.from_data_list([data])
        # batch = batch.to(self.device)
        # batch.x = batch.mask.reshape(len(batch.mask), 1)
        self.eval()
        with torch.no_grad():
            # trajectory = self.model.faster_predict_step(batch, 0)
            trajectory = self.model.predict_step(batch, 0)
        # print("Trajectory prediction shape:", trajectory.shape)
        return trajectory.cpu()
