# TODO: merge uneven object dataset collection code
import os
import json
import pickle as pkl
import hydra
import lightning as L
import omegaconf

# Modules
import rpad.pyg.nets.pointnet2 as pnp_orig
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from have.generator.datasets.uneven_object import UnevenObjectDataModule
from have.generator.datasets.flow_trajectory import FlowTrajectoryDataModule
from have.generator.models.flow_diffuser_pndit import (   # For unconditional diffusion training
    FlowTrajectoryDiffusionModule_PNDiT,
)
from have.generator.nets.dit_models import (
    PN2DiT,    # For unconditional diffusion training
)
from have.generator.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    match_fn,
)

data_module_class = {
    "uneven": UnevenObjectDataModule,
    "articulated": FlowTrajectoryDataModule
}
training_module_class = FlowTrajectoryDiffusionModule_PNDiT  # PNDiT (Uncond diffusion)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    # torch.set_float32_matmul_precision("medium")
    torch.set_float32_matmul_precision("highest")

    # Global seed for reproducibility.
    L.seed_everything(cfg.seed)

    ######################################################################
    # Create the datamodule.
    # The datamodule is responsible for all the data loading, including
    # downloading the data, and splitting it into train/val/test.
    #
    # This could be swapped out for a different datamodule in-place,
    # or with an if statement, or by using hydra.instantiate.
    ######################################################################

    toy_dataset = None
    if cfg.dataset.toy_dataset_path is not None:
        with open(cfg.dataset.toy_dataset_path, 'r') as f:
            toy_dataset = json.load(f)

    # Create flow dataset
    datamodule = data_module_class[cfg.dataset.task](
        root=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.resources.num_workers,  #0
        n_proc=cfg.resources.n_proc_per_worker,  #1
        seed=cfg.seed,
        history=False,   # Only support unconditional generator in this repo
        randomize_size=cfg.dataset.randomize_size,
        augmentation=cfg.dataset.augmentation,
        trajectory_len=1,  # Only used when training model that predicts trajectories
        special_req=cfg.dataset.special_req,
        n_repeat=200,
        toy_dataset=toy_dataset,
    )
    train_loader = datamodule.train_dataloader()
    
    cfg.training.train_sample_number = len(train_loader)
    eval_sample_bsz = 1 if cfg.training.wta else cfg.training.batch_size
    train_val_loader = datamodule.train_val_dataloader(bsz=eval_sample_bsz)

    if 'door' in cfg.dataset.name:  # multimodal-door dataset
        # For half-half training:
        # - Unseen loader: randomly opened doors
        # - Validation loader: fully closed doors
        fully_closed_datamodule = data_module_class[cfg.dataset.task](
            root=cfg.dataset.data_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.resources.num_workers,
            n_proc=cfg.resources.n_proc_per_worker,
            seed=cfg.seed,
            history="his" in cfg.model.name,
            randomize_size=cfg.dataset.randomize_size,
            augmentation=cfg.dataset.augmentation,
            trajectory_len=1,  # Only used when training trajectory model
            special_req="fully-closed",  # special_req="fully-closed"
            toy_dataset=toy_dataset,
        )
        val_loader = datamodule.val_dataloader(bsz=eval_sample_bsz) #fully_closed_datamodule.val_dataloader(bsz=eval_sample_bsz)
        unseen_loader = fully_closed_datamodule.unseen_dataloader(
            bsz=eval_sample_bsz
        )
    else:  # half-half full dataset
        val_loader = datamodule.val_dataloader(bsz=eval_sample_bsz)
        unseen_loader = datamodule.unseen_dataloader(bsz=eval_sample_bsz)


    ######################################################################
    # Create the network(s) which will be trained by the Training Module.
    # The network should (ideally) be lightning-independent. This allows
    # us to use the network in other projects, or in other training
    # configurations.
    #
    # This might get a bit more complicated if we have multiple networks,
    # but we can just customize the training module and the Hydra configs
    # to handle that case. No need to over-engineer it. You might
    # want to put this into a "create_network" function somewhere so train
    # and eval can be the same.
    #
    # If it's a custom network, a good idea is to put the custom network
    # in `have.generator.nets.my_net`.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).

    # Only support pndit in this repo for now
    network = PN2DiT(
        in_channels=3,
        depth=5,
        hidden_size=128,
        patch_size=1,
        num_heads=4,
        n_points=cfg.dataset.n_points,
    )
        

    ######################################################################
    # Create the training module.
    # The training module is responsible for all the different parts of
    # training, including the network, the optimizer, the loss function,
    # and the logging.
    ######################################################################

    model = training_module_class(
        network, training_cfg=cfg.training, model_cfg=cfg.model
    )

    ######################################################################
    # Set up logging in WandB.
    # This is a bit complicated, because we want to log the codebase,
    # the model, and the checkpoints.
    ######################################################################

    # If no group is provided, then we should create a new one (so we can allocate)
    # evaluations to this group later.
    if cfg.wandb.group is None:
        id = wandb.util.generate_id()
        group = "experiment-" + id
    else:
        group = cfg.wandb.group

    logger = WandbLogger(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        log_model=True,  # Only log the last checkpoint to wandb, and only the LAST model checkpoint.
        save_dir=cfg.wandb.save_dir,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        job_type=cfg.job_type,
        save_code=True,  # This just has the main script.
        group=group,
    )

    ######################################################################
    # Create the trainer.
    # The trainer is responsible for running the training loop, and
    # logging the results.
    #
    # There are a few callbacks (which we could customize):
    # - LogPredictionSamplesCallback: Logs some examples from the dataset,
    #       and the model's predictions.
    # - ModelCheckpoint #1: Saves the latest model.
    # - ModelCheckpoint #2: Saves the best model (according to validation
    #       loss), and logs it to wandb.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        max_epochs=cfg.training.epochs,
        logger=logger,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        callbacks=[
            # Callback which logs whatever visuals (i.e. dataset examples, preds, etc.) we want.
            LogPredictionSamplesCallback(
                logger=logger,
                eval_per_n_epoch=cfg.training.check_val_every_n_epoch,
                eval_dataloader_lengths=[
                    len(val_loader),
                    # len(multimodal_loader) if "his" in cfg.model.name else len(unseen_loader),
                    # len(train_val_loader),
                    len(unseen_loader),
                ],
                eval_dataloader_names=['val', 'unseen'], 
            ),
            # This checkpoint callback saves the latest model during training, i.e. so we can resume if it crashes.
            # It saves everything, and you can load by referencing last.ckpt.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}",
                monitor="step",
                mode="max",
                save_weights_only=False,
                save_last=True,
            ),
            # This checkpoint will get saved to WandB. The Callback mechanism in lightning is poorly designed, so we have to put it last.
            ModelCheckpoint(
                dirpath=cfg.lightning.checkpoint_dir,
                filename="{epoch}-{step}-{val_loss:.2f}-weights-only",
                monitor="val_wta/rmse" if cfg.training.wta else "val/rmse",
                mode="min",
                save_weights_only=True,
            ),
        ],
    )

    ######################################################################
    # Log the code to wandb.
    # This is somewhat custom, you'll have to edit this to include whatever
    # additional files you want, but basically it just logs all the files
    # in the project root inside dirs, and with extensions.
    ######################################################################

    # Log the code used to train the model. Make sure not to log too much, because it will be too big.
    wandb.run.log_code(
        root=PROJECT_ROOT,
        include_fn=match_fn(
            dirs=["configs", "scripts", "src"],
            extensions=[".py", ".yaml"],
        ),
    )

    ######################################################################
    # Train the model.
    ######################################################################

    model.set_dataloader_names(["val", "unseen"])
    trainer.fit(model, train_loader, [val_loader, unseen_loader], ckpt_path=None)


if __name__ == "__main__":
    main()
