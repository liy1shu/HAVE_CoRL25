import os

import lightning as L
import rpad.partnet_mobility_utils.dataset as rpd
import torch_geometric.loader as tgl
from rpad.pyg.dataset import CachedByKeyDataset

from flowbothd.datasets.uneven_object_dataset_pyg import (
    UnevenObjectPyGDataset,
    UnevenObjectHistoryDataset
)


# Create UnevenObject datamodule
class UnevenObjectDataModule(L.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        n_proc,
        history=False,  ## With / without history # TODO: not implemented yet
        randomize_camera: bool = True,
        randomize_size: bool = False,
        augmentation: bool = False,
        trajectory_len: int = 1,
        seed: int = 42,
        special_req: str = None,
        toy_dataset: dict = None,
        n_repeat: int = 100,  # By default, repeat training dataset by 100
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_cls = UnevenObjectHistoryDataset if history else UnevenObjectPyGDataset
        if augmentation:  # Augmentation: 4 flip modes
            n_repeat *= 4
        print(
            self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"]
            )
        )
        self.train_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/train"),
                split="umpnet-train-train"
                if toy_dataset is None
                else toy_dataset["train-train"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-train"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=n_repeat,
            # n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.train_val_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/train"),
                split="umpnet-train-train"
                if toy_dataset is None
                else toy_dataset["train-train"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-train"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.val_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/val"),
                split="umpnet-train-test"
                if toy_dataset is None
                else toy_dataset["train-test"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TEST_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-test"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.unseen_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/test"),
                split="umpnet-test" if toy_dataset is None else toy_dataset["test"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TEST_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["test"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

    def train_dataloader(self):
        L.seed_everything(self.seed)
        return tgl.DataLoader(
            self.train_dset, self.batch_size, shuffle=True, num_workers=0
        )

    def train_val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.train_val_dset,
            bsz,
            shuffle=False,
            num_workers=0
            # self.train_val_dset, 1, shuffle=False, num_workers=0
        )

    def val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.val_dset,
            bsz,
            shuffle=False,
            num_workers=0
            # self.val_dset, 1, shuffle=False, num_workers=0
        )

    def unseen_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.unseen_dset,
            # self.batch_size,
            bsz,
            shuffle=False,
            num_workers=0
            # self.unseen_dset, self.batch_size, shuffle=False, num_workers=0
        )

# Create UnevenObject datamodule
class UnevenObjectToyDataModule(L.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        n_proc,
        history=False,  ## With / without history # TODO: not implemented yet
        randomize_camera: bool = True,
        randomize_size: bool = False,
        augmentation: bool = False,
        trajectory_len: int = 1,
        seed: int = 42,
        special_req: str = None,
        toy_dataset: dict = None,
        n_repeat: int = 100,  # By default, repeat training dataset by 100
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_cls = UnevenObjectHistoryDataset if history else UnevenObjectPyGDataset
        if augmentation:  # Augmentation: 4 flip modes
            n_repeat *= 4
        print(
            self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"]
            )
        )
        self.train_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/toy"),
                split="umpnet-train-train"
                if toy_dataset is None
                else toy_dataset["train-train"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-train"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=n_repeat,
            # n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.train_val_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/toy"),
                split="umpnet-train-train"
                if toy_dataset is None
                else toy_dataset["train-train"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-train"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.val_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/toy"),
                split="umpnet-train-test"
                if toy_dataset is None
                else toy_dataset["train-test"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TRAIN_TEST_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["train-test"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.unseen_dset = CachedByKeyDataset(
            dset_cls=self.dataset_cls,
            dset_kwargs=dict(
                root=os.path.join(root, "raw/toy"),
                split="umpnet-test" if toy_dataset is None else toy_dataset["test"],
                randomize_camera=randomize_camera,
                randomize_size=randomize_size,
                augmentation=augmentation,
                trajectory_len=trajectory_len,
                special_req=special_req,
            ),
            data_keys=rpd.UMPNET_TEST_OBJ_IDS
            if toy_dataset is None
            else toy_dataset["test"],
            root=root,
            processed_dirname=self.dataset_cls.get_processed_dir(
                # True,
                randomize_camera,
                trajectory_len,
                special_req,
                randomize_size=randomize_size,
                augmentation=augmentation,
                toy_dataset_id=None if toy_dataset is None else toy_dataset["id"],
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

    def train_dataloader(self):
        L.seed_everything(self.seed)
        return tgl.DataLoader(
            self.train_dset, self.batch_size, shuffle=True, num_workers=0
        )

    def train_val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.train_val_dset,
            bsz,
            shuffle=False,
            num_workers=0
            # self.train_val_dset, 1, shuffle=False, num_workers=0
        )

    def val_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.val_dset,
            bsz,
            shuffle=False,
            num_workers=0
            # self.val_dset, 1, shuffle=False, num_workers=0
        )

    def unseen_dataloader(self, bsz=None):
        bsz = self.batch_size if bsz is None else bsz
        return tgl.DataLoader(
            self.unseen_dset,
            # self.batch_size,
            bsz,
            shuffle=False,
            num_workers=0
            # self.unseen_dset, self.batch_size, shuffle=False, num_workers=0
        )
