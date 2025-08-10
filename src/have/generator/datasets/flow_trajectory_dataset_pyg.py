from typing import List, Optional, Protocol, Union, cast

import rpad.partnet_mobility_utils.dataset as pmd
import torch
import torch_geometric.data as tgd

from have.generator.datasets.flow_trajectory_dataset import (
    FlowTrajectoryDataset,
)


class FlowTrajectoryTGData(Protocol):
    id: str  # Object ID.

    pos: torch.Tensor  # Points in the point cloud.
    delta: torch.Tensor  # instantaneous positive 3D flow trajectories.
    point: torch.Tensor  # the trajectory waypoints
    mask: torch.Tensor  # Mask of the part of interest.


class FlowTrajectoryPyGDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        randomize_joints: bool = True,  # TODO: set to True
        randomize_camera: bool = True,
        randomize_size: bool = False,
        augmentation: bool = False,
        trajectory_len: int = 5,
        special_req: str = None,
        n_points: Optional[int] = 1200,
        seed: int = 42,  # Randomize everything
    ) -> None:
        super().__init__()
        self.dataset = FlowTrajectoryDataset(
            root,
            split,
            randomize_joints,
            randomize_camera,
            trajectory_len,
            special_req,
            n_points,
        )
        self.n_points = n_points
        self.seed = seed
        self.randomize_size = randomize_size
        self.augmentation = augmentation

    def len(self) -> int:
        return len(self.dataset)

    def get(self, index) -> tgd.Data:
        return self.get_data(self.dataset._dataset._ids[index], seed=self.seed)

    @staticmethod
    def get_processed_dir(
        randomize_joints,
        randomize_camera,
        trajectory_len,
        special_req=None,
        toy_dataset_id=None,
        randomize_size: bool = False,
        augmentation: bool = False,
    ):
        joint_chunk = "rj" if randomize_joints else "sj"
        camera_chunk = "rc" if randomize_camera else "sc"
        random_size_str = "" if not randomize_size else "_rsz"
        augmentation_str = "" if not augmentation else "_aug"
        if special_req is None and toy_dataset_id is None:
            return f"processed_{trajectory_len}_{joint_chunk}_{camera_chunk}_random{random_size_str}{augmentation_str}"
        elif special_req is not None and toy_dataset_id is None:
            # fully_closed
            # half_half
            return f"processed_{trajectory_len}_{joint_chunk}_{camera_chunk}_{special_req}{random_size_str}{augmentation_str}"
        elif special_req is None and toy_dataset_id is not None:
            # fully_closed
            # half_half
            return f"processed_{trajectory_len}_{joint_chunk}_{camera_chunk}_toy{toy_dataset_id}_random{random_size_str}{augmentation_str}"
        else:
            return f"processed_{trajectory_len}_{joint_chunk}_{camera_chunk}_{special_req}_toy{toy_dataset_id}{random_size_str}{augmentation_str}"

    def get_data(self, obj_id: str, seed) -> FlowTrajectoryTGData:
        data_dict = self.dataset.get_data(obj_id, seed)
        # random size
        rsz = 1 if not self.randomize_size else np.random.uniform(0.1, 5)
        # data augmentation
        flip = 0 if not self.augmentation else np.random.randint(0, 4)  # 4 flip modes
        flip_mat = torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Normal
                [[1, 0, 0], [0, -1, 0], [0, 0, 1]],  # Left, right
                [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Front, back
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # Front back & left right
            ]
        ).float()

        data = tgd.Data(
            id=data_dict["id"],
            pos=torch.matmul(
                torch.from_numpy(data_dict["pos"]).float() * rsz, flip_mat[flip]
            ),
            delta=torch.matmul(
                torch.from_numpy(data_dict["delta"]).float(), flip_mat[flip]
            ),
            point=torch.matmul(
                torch.from_numpy(data_dict["point"]).float() * rsz, flip_mat[flip]
            ),
            mask=torch.from_numpy(data_dict["mask"]).float(),
        )
        return cast(FlowTrajectoryTGData, data)
