import copy
from typing import Dict, List, Literal, Optional, Sequence, TypedDict, Union, Tuple

import numpy as np
import numpy.typing as npt
import rpad.partnet_mobility_utils.articulate as pma
import rpad.partnet_mobility_utils.dataset as pmd
# from rpad.partnet_mobility_utils.data import PMObject
from pathlib import Path
from rpad.pybullet_libs.utils import isnotebook, suppress_stdout
from rpad.core.distributed import NPSeed
import pybullet as p
import pybullet_data
import os
import math
import xml.etree.ElementTree as ET
from rpad.pybullet_libs.camera import Camera
from rpad.partnet_mobility_utils.render.pybullet import sample_az_ele

class UnevenObjectData(TypedDict):
    id: str
    pos: npt.NDArray[np.float32]  # (N, 3): Point cloud observation.
    delta: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth flow.
    point: npt.NDArray[np.float32]  # (N, K, traj_len * 3): Ground-truth waypoints.
    mask: npt.NDArray[np.bool_]  #  (N,): Mask the point of interest.

class UnevenPC(TypedDict):
    """A Partial PointCloud for uneven objetc

    Attributes:
        pos: Position
        seg: segmentation
    """

    id: str
    pos: npt.NDArray[np.float32]
    P_world_rod: npt.NDArray[np.float32]
    seg: npt.NDArray[np.uint]
    frame: Literal["world", "camera"]
    T_world_cam: npt.NDArray[np.float32]
    T_world_base: npt.NDArray[np.float32]
    # proj_matrix: npt.NDArray[np.float32]
    labelmap: Dict[str, int]
    # angles: Dict[str, float]
    center_of_mass: npt.NDArray[np.float32]
    segmap: npt.NDArray[np.uint]

"""
Changes made:
Apart from flow, also return new pos (P_world_new) and new joint angles (target_jas)
"""

def compute_flow(
    P_world: npt.NDArray[np.float32],
    T_world_base: npt.NDArray[np.float32],
    center_of_mass: npt.NDArray[np.float32],
    pc_seg_obj: npt.NDArray[np.uint8],
    option: str,
) -> npt.NDArray[np.float32]:
    """Compute normalized flow for an object, based on its kinematics.

    Args:
        P_world (npt.NDArray[np.float32]): Point cloud render of the object in the world frame.
        T_world_base (npt.NDArray[np.float32]): The pose of the base link in the world frame.
        # current_jas (Dict[str, float]): The current joint angles (easy to acquire from the render that created the points.)
        pc_seg (npt.NDArray[np.uint8]): The segmentation labels of each point.
        labelmap (Dict[str, int]): Map from the link name to segmentation name.
        pm_raw_data (PMObject): The object description, essentially providing the kinematic structure of the object.
        # linknames (Union[Literal['all'], Sequence[str]], optional): The names of the links for which compute flow. Defaults to "all", which will articulate all of them.

    Returns:
        npt.NDArray[np.float32]: _description_
    """
    P_world_rod = P_world.copy()
    # P_world_rod[pc_seg_obj!=1] = 0

    # distance to x
    x_center_of_mass = [0, center_of_mass[1], 0]
    distance = (P_world_rod - x_center_of_mass)[:,1]
    
    if option == "centerOnly":
        min_distance_index = np.argmin(abs(distance))
        flow_dim1 = np.ones_like(distance)
        flow_dim1[min_distance_index] = 0
    elif option == "torque":
        flow_dim1 = abs(distance)
    elif option == "rotateMovement":
        flow_dim1 = distance
    flow = np.zeros_like(P_world_rod)
    flow[:, 2] = flow_dim1 / abs(flow_dim1[np.argmax(abs(flow_dim1))])
    
    P_world_new = P_world_rod.copy()+flow
    P_world_new[pc_seg_obj!=1] = 0
    
    return P_world_new, flow


# Compute trajectories as K deltas & waypoints
def compute_flow_trajectory( # the flow and p_world_new given a small perturbation
    K,
    P_world,
    T_world_base,
    center_of_mass,
    # current_jas,
    pc_seg_obj,
    # labelmap,
    # pm_raw_data,
    # linknames="all",
    option,
) -> npt.NDArray[np.float32]:
    flow_trajectory = np.zeros((K, P_world.shape[0], 3), dtype=np.float32)
    point_trajectory = np.zeros((K, P_world.shape[0], 3), dtype=np.float32)
    for step in range(K):
        # compute the delta / waypoint & rotate and then calculate another
        P_world_new, flow = compute_flow(
            P_world,
            T_world_base,
            center_of_mass,
            # current_jas,
            pc_seg_obj,
            # labelmap,
            # pm_raw_data,
            # linknames,
            option,
        )
        flow_trajectory[step, :, :] = flow
        point_trajectory[step, :, :] = P_world_new
        # # Update pos
        # P_world = P_world_new
    return flow_trajectory.transpose(1, 0, 2), point_trajectory.transpose(
        1, 0, 2
    )  # Delta / Point * traj_len * 3

AVAILABLE_DATASET = Literal[
    "all", "umpnet-train-train", "umpnet-train-test", "umpnet-test"
]

class UnevenObjectDataset:
    def __init__(
        self,
        root: str,
        split: Union[pmd.AVAILABLE_DATASET, List[str]],
        # randomize_joints: bool = True,
        randomize_camera: bool = True,
        trajectory_len: int = 1,
        special_req: str = None,
        n_points: Optional[int] = None,
    ) -> None:
        """The FlowBot3D dataset. Set n_points depending if you can handle ragged batches or not.

        Args:
            root (str): The root directory of the downloaded partnet-mobility dataset.
            split (Union[pmd.AVAILABLE_DATASET, List[str]]): Either an available split like "umpnet-train-train" or a list of object IDs from the PM dataset.
            # randomize_joints (bool): Whether or not to randomize the joints.
            randomize_camera (bool): Whether or not to randomize the camera location (in a fixed range, see the underlying renderer...)
            n_points (Optional[int], optional): Whether or not to downsample the number of points returned for each example. If
                you want to use this datasets as a standard PyTorch dataset, you should set this to a non-None value (otherwise passing it into
                a dataloader won't really work, since you'll have ragged batches. If you're using PyTorch-Geometric to handle batches, do whatever you want.
                Defaults to None.
        """
        self._dataset = UOPCDataset(root=root, split=split, renderer="pybullet")
        self._ids = self._dataset._ids
        # self.randomize_joints = randomize_joints
        self.randomize_camera = randomize_camera
        self.trajectory_len = trajectory_len
        self.special_req = special_req
        self.n_points = n_points

    def get_data(self, obj_id: str, seed=None) -> UnevenObjectData:
        # if self.special_req is None:
        #     joints = "random" if self.randomize_joints else None
        # else:
        #     joints = (
        #         self.special_req
        #     )  # past-todo: Set to random-oc as for multimodal experiments
        # # print(joints)
        # # joints = "random" if self.randomize_joints else None

        # Select the camera.
        camera_xyz = "random" if self.randomize_camera else None

        rng = np.random.default_rng(seed)
        seed1, seed2 = rng.bit_generator._seed_seq.spawn(2)  # type: ignore

        data = self._dataset.get(
            obj_id=obj_id, 
            # joints=joints, 
            camera_xyz=camera_xyz, 
            seed=seed1  # type: ignore
        )
        pos = data["pos"]
        # pos = data["normalized_pc"]

        # Compute the flow trajectory
        flow_trajectory, point_trajectory = compute_flow_trajectory(
            K=self.trajectory_len,
            P_world=pos,
            T_world_base=data["T_world_base"],
            center_of_mass=data["center_of_mass"],
            # center_of_mass=data["normalize_com"],
            # current_jas=data["angles"],
            pc_seg_obj=data["pc_seg_obj"],
            # labelmap=data["labelmap"],
            # pm_raw_data=self._dataset.pm_objs[obj_id],
            # linknames="all",
            option=self.special_req, # "centerOnly" for gt-1 other-0 and "torque/rotateMovement" for (abs) distance to the center of mass
        )
        # Compute the mask of any part which has flow.
        mask = (
            ~(
                np.isclose(flow_trajectory.reshape(flow_trajectory.shape[0], -1), 0.0)
            ).all(axis=-1)
        ).astype(np.bool_)
        
        if self.n_points:
            if len(flow_trajectory) < self.n_points:
                repeat_times = self.n_points // len(flow_trajectory) + 1
                pos = np.tile(pos, (repeat_times, 1))
                flow_trajectory = np.tile(flow_trajectory, (repeat_times, 1, 1))
                point_trajectory = np.tile(point_trajectory, (repeat_times, 1, 1))
                mask = np.tile(mask, (repeat_times))
            
            
            rng = np.random.default_rng(seed2)
            ixs = rng.permutation(range(len(flow_trajectory)))[: self.n_points]
            pos = pos[ixs]
            flow_trajectory = flow_trajectory[ixs, :, :]
            point_trajectory = point_trajectory[ixs, :, :]
            mask = mask[ixs]
            
        return {
            "id": data["id"],
            "pos": pos,
            "delta": flow_trajectory,  #  N , traj_len, 3
            "point": point_trajectory,  #  N , traj_len, 3
            "mask": mask,
        }

    def __getitem__(self, item: int) -> UnevenObjectData:
        obj_id = self._dataset._ids[item]
        return self.get_data(obj_id)

    def __len__(self):
        return len(self._dataset)

DEFAULT_OBJS = [str(i) for i in range(240)]
DEFAULT_OBJS.extend(["bookmark1", "knife", "bookmark2", "chopsticks"])

class UOPCDataset:
    """Uneven Object Point Cloud Dataset"""

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[AVAILABLE_DATASET, List[str]],
        renderer: Literal["pybullet", "sapien", "trimesh"] = "pybullet",
    ):
        if isinstance(split, str):
            if split == "all":
                self._ids = DEFAULT_OBJS
            else:
                # TODO: implement
                # ids = read_ids(
                #     {
                #         "umpnet-train-train": UMPNET_DIR / "train_train.txt",
                #         "umpnet-train-test": UMPNET_DIR / "train_test.txt",
                #         "umpnet-test": UMPNET_DIR / "test.txt",
                #     }[split]
                # )
                # self._ids = [id[0] for id in ids]
                raise
        else:
            self._ids = copy.deepcopy(split)

        new_ids = []
        def_objs = set(DEFAULT_OBJS)
        for id in self._ids:
            if id not in def_objs:
                logging.warning(f"{id} is not well-formed, excluding...")
                raise ValueError("BDADAD")
            else:
                new_ids.append(id)

        self._ids = new_ids # this is ids of all objs used, default ones \cap split-mentioned ones

        self.uneven_objs: Dict[str, Path(obj_dir)] = {
            id: (Path(root) / id) for id in self._ids
        }
        self.renderers: Dict[str, UnevenRenderer] = {}
        self.renderer_type = renderer

    def get(
        self,
        obj_id: str,
        # joints: Union[
        #     Literal["random"],
        #     Dict[str, Union[float, Literal["random", "random-oc"]]],
        #     None,
        # ] = None,
        camera_xyz: Union[
            Literal["random"],
            Tuple[float, float, float],
            None,
        ] = None,
        seed: Optional[int] = None,
    ) -> UnevenPC:
        if obj_id not in self.renderers:
            if self.renderer_type == "pybullet":
                new_renderer = UnevenRenderer(obj_id, str(self.uneven_objs[obj_id].parent))
            else:
                raise NotImplementedError("not yet implemented")
            self.renderers[obj_id] = new_renderer
        renderer = self.renderers[obj_id]

        pc_render = renderer.render(
            uneven_obj=self.uneven_objs[obj_id], # this is a PosixPath
            # joints=joints,
            camera_xyz=camera_xyz,
            seed=seed,
        )

        return pc_render

    def __getitem__(self, item: Union[int, str]) -> UnevenPC:
        if isinstance(item, str):
            obj_id = item
        else:
            obj_id = self._ids[item]

        return self.get(
            obj_id, 
            # joints=None, 
            camera_xyz=None)

    def __len__(self):
        return len(self._ids)

class UnevenRenderer():
    def __init__(
        self,
        obj_id: str,
        dataset_path: str,
        camera_pos: List = [1, 1, 1],
        gui: bool = False,
        with_plane: bool = True,
    ):
        self.with_plane = with_plane
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        # Add in a plane.
        if with_plane:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        # Add in gravity.
        p.setGravity(0, 0, 0, self.client_id)

        # Add in the object.
        self.obj_id_str = obj_id
        self.rodPath = os.path.join(dataset_path, obj_id, "rod.urdf")

        # load cuboid and spatula
        self.rodShape, self.rodMassPosition = self._get_urdf_shape(self.rodPath)
        self.cuboidPath = "cuboid.urdf"
        self.cuboidShape, _ = self._get_urdf_shape(self.cuboidPath)
        self.rodPosition = [0 + self.rodMassPosition[1], 
                            0 + self.rodMassPosition[0], 
                            self.cuboidShape[2] + self.rodShape[2]/2]
        self.rodOrientation = p.getQuaternionFromEuler([0, 0, math.radians(90)])
        if isnotebook() or "PYTEST_CURRENT_TEST" in os.environ:
            self.obj_id = p.loadURDF(
                self.rodPath,
                basePosition=self.rodPosition,
                baseOrientation=self.rodOrientation,
                useFixedBase=True,
                # flags=p.URDF_MAINTAIN_LINK_ORDER,
                physicsClientId=self.client_id,
            )
            self.supporting_cuboid_1 = p.loadURDF(
                self.cuboidPath,
                basePosition = [self.rodPosition[0],  (self.rodShape[0]/2)*0.9, 0],
                useFixedBase=True,
                physicsClientId=self.client_id
            )
            self.supporting_cuboid_2 = p.loadURDF(
                self.cuboidPath,
                basePosition = [self.rodPosition[0], -(self.rodShape[0]/2)*0.9, 0],
                useFixedBase=True,
                physicsClientId=self.client_id
            )

        else:
            with suppress_stdout():
                self.obj_id = p.loadURDF(
                    self.rodPath,
                    basePosition=self.rodPosition,
                    baseOrientation=self.rodOrientation,
                    useFixedBase=True,
                    # flags=p.URDF_MAINTAIN_LINK_ORDER,
                    physicsClientId=self.client_id,
                )
                self.supporting_cuboid_1 = p.loadURDF(
                    self.cuboidPath,
                    basePosition = [self.rodPosition[0],  (self.rodShape[0]/2)*0.9, self.cuboidShape[2]/2],
                    useFixedBase=True,
                    physicsClientId=self.client_id
                )
                self.supporting_cuboid_2 = p.loadURDF(
                    self.cuboidPath,
                    basePosition = [self.rodPosition[0], -(self.rodShape[0]/2)*0.9, self.cuboidShape[2]/2],
                    useFixedBase=True,
                    physicsClientId=self.client_id
                )
                
        # The object isn't placed at the bottom of the scene.
        xyz_offset = p.getAABB(self.obj_id, physicsClientId=self.client_id)
        self.x_offset = (xyz_offset[1][0] - xyz_offset[0][0])/2
        self.y_offset = (xyz_offset[1][1] - xyz_offset[0][1])/2
        self.z_offset = (xyz_offset[1][2] - xyz_offset[0][2])/2
        p.resetBasePositionAndOrientation(
            self.obj_id,
            posObj=[self.rodPosition[0],
                    self.rodPosition[1],
                    self.cuboidShape[2] + self.z_offset],
            ornObj=self.rodOrientation,
            physicsClientId=self.client_id,
        )
        p.resetBasePositionAndOrientation(
            self.supporting_cuboid_1,
            posObj=[self.rodPosition[0],
                    (self.y_offset)*0.9, 
                    self.cuboidShape[2]/2],
            ornObj=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client_id,
        )
        p.resetBasePositionAndOrientation(
            self.supporting_cuboid_2,
            posObj=[self.rodPosition[0],
                    -(self.y_offset)*0.9, 
                    self.cuboidShape[2]/2],
            ornObj=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client_id,
        )
        self.T_world_base = np.eye(4)
        self.T_world_base[2, 3] = self.cuboidShape[2]-self.z_offset

        # Create a camera.
        # self.camera = Camera(pos=camera_pos, znear=0.01, zfar=10)
        
        self.init_camera()

        # From https://pybullet.org/Bullet/phpBB3/viewtopic.php?f=24&t=12728&p=42293&hilit=linkIndex#p42293
        self.link_name_to_index = {
            p.getBodyInfo(self.obj_id, physicsClientId=self.client_id)[0].decode(
                "UTF-8"
            ): -1,
        }
        self.jn_to_ix = {}

        # Get the segmentation.
        for _id in range(p.getNumJoints(self.obj_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.obj_id, _id, physicsClientId=self.client_id)
            joint_name = info[1].decode("UTF-8")
            link_name = info[12].decode("UTF-8")
            self.link_name_to_index[link_name] = _id

            # Only store if the joint is one we can control.
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                self.jn_to_ix[joint_name] = _id


    def init_camera(self, camera_eye=[1.5, 0, 0.5], camera_target=[0, 0, 0], up_vector = [0, 0, 1], fov=60, frame_width=640, frame_height=480, near=0.1, far=100):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_eye = camera_eye
        self.camera_target = camera_target
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, up_vector)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov, frame_width / frame_height, near, far)
        self.camera = Camera(
            pos=camera_eye, 
            render_height=self.frame_height,
            render_width=self.frame_width,
            znear=near,
            zfar=far,
            # intrinsics=DEFAULT_CAMERA_INTRINSICS,
            target=self.camera_target
        )

    def render(
        self,
        uneven_obj: Path,
        camera_xyz: Union[
            Literal["random"],
            Tuple[float, float, float],
            None,
        ] = None,
        seed: Optional[int] = None,
    ) -> UnevenPC:
        rng = np.random.default_rng(seed)
        _, seed = rng.bit_generator._seed_seq.spawn(2)  # type: ignore
        
        if camera_xyz is not None:
            self.set_camera(camera_xyz, seed=seed)
            
        obs = self.camera.render(self.client_id, self.with_plane) # link seg
        
        remove_cuboid=True
        if remove_cuboid:
            idx=(obs["P_world"][:,2]>=0.065)
            obs["P_world"] = obs["P_world"][idx]
            obs["pc_seg"] = obs["pc_seg"][idx]
            
        rgb = obs["rgb"]
        depth = obs["depth"]
        seg = obs["seg"]
        P_cam = obs["P_cam"]
        P_world = obs["P_world"]
        pc_seg = obs["pc_seg"]
        segmap = obs["segmap"]
        
        # Reindex the segmentation.
        pc_seg_obj = np.ones_like(pc_seg) * 0
        for k, (body, link) in segmap.items():
            if body == self.obj_id:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = k

        P_world_rod = P_world[pc_seg_obj==1]
        
        # normalized_pc, scale_factors, pcd_center, normalize_com = self.normalize_pcd(P_world, P_world_rod)
        
        return {
            "id": uneven_obj.name,
            "pos": P_world,
            "P_world_rod": P_world_rod,
            "seg": seg,
            "pc_seg_obj": pc_seg_obj,
            "frame": "world",
            "T_world_cam": self.camera.T_world2cam,
            "T_world_base": np.copy(self.T_world_base),
            # "proj_matrix": None,
            "labelmap": self.link_name_to_index,
            # "angles": self._render_env.get_joint_angles(),
            "center_of_mass": self.get_center_of_mass(P_world_rod),
            "segmap": segmap,
            # "normalized_pc": normalized_pc,
            # "normalize_com": normalize_com,
        }
        
    def set_camera(
        self,
        camera_xyz: Union[Literal["random"], Tuple[float, float, float]],
        seed: Optional[NPSeed] = None,
    ):
        if camera_xyz == "random":
            x, y, z, az, el = sample_az_ele(
                np.sqrt(8),
                np.deg2rad(30),
                np.deg2rad(150),
                np.deg2rad(30),
                np.deg2rad(60),
                seed=seed,
            )
            camera_xyz = (x, y, z)

        self.camera.set_camera_position(camera_xyz)
        
    def close(self):
        p.disconnect(self.client_id)
    
    def get_center_of_mass(self, P_world_rod=None):
        # obj size here represents for the ground truth size of the object
        obj_size, center_of_mass_rod = self._get_urdf_shape(self.rodPath)
        # base position here represents for the center of mass ground truth position
        base_position, base_orientation = p.getBasePositionAndOrientation(self.obj_id)
        
        # P_world_rod not given, return ground truth center of mass
        if P_world_rod is None:
            return base_position

        # P_world_rod given, project the ground truth center of mass to pcd center of mass
        xmax = np.max(P_world_rod[:, 0])
        xmin = np.min(P_world_rod[:, 0])
        ymax = np.max(P_world_rod[:, 1])
        ymin = np.min(P_world_rod[:, 1])
        zmax = np.max(P_world_rod[:, 2])
        zmin = np.min(P_world_rod[:, 2])
        
        y_true, x_true, z_true = obj_size
        y_mass, x_mass, z_mass = center_of_mass_rod

        center_of_mass_pcd = (
            ((xmax-xmin)/x_true)*x_mass+(xmax+xmin)/2,
            ((ymax-ymin)/y_true)*y_mass+(ymax+ymin)/2,
            ((zmax-zmin)/z_true)*z_mass+(zmax+zmin)/2,
        )

        return center_of_mass_pcd
    
    def normalize_pcd(self, P_world_rod_and_cuboid, P_world_rod):
        # obj size here represents for the ground truth size of the object
        obj_size, center_of_mass_rod = self._get_urdf_shape(self.rodPath)
        # base position here represents for the center of mass ground truth position
        base_position, base_orientation = p.getBasePositionAndOrientation(self.obj_id)

        # P_world_rod given, project the ground truth center of mass to pcd center of mass
        xmax = np.max(P_world_rod[:, 0])
        xmin = np.min(P_world_rod[:, 0])
        ymax = np.max(P_world_rod[:, 1])
        ymin = np.min(P_world_rod[:, 1])
        zmax = np.max(P_world_rod[:, 2])
        zmin = np.min(P_world_rod[:, 2])
        
        y_true, x_true, z_true = obj_size
        y_mass, x_mass, z_mass = center_of_mass_rod

        center_of_mass_pcd = (
            ((xmax-xmin)/x_true)*x_mass + (xmax + xmin)/2,
            ((ymax-ymin)/y_true)*y_mass + (ymax + ymin)/2,
            ((zmax-zmin)/z_true)*z_mass + (zmax + zmin)/2,
        )

        xmax = np.max(P_world_rod_and_cuboid[:, 0])
        xmin = np.min(P_world_rod_and_cuboid[:, 0])
        ymax = np.max(P_world_rod_and_cuboid[:, 1])
        ymin = np.min(P_world_rod_and_cuboid[:, 1])
        zmax = np.max(P_world_rod_and_cuboid[:, 2])
        zmin = np.min(P_world_rod_and_cuboid[:, 2])
        
        # 1. Center the point cloud (move to origin)
        pcd_center = np.array([(xmax + xmin)/2, (ymax + ymin)/2, (zmax + zmin)/2])
        centered_pc = P_world_rod_and_cuboid - pcd_center
        
        # 2. Calculate current ranges (before scaling)
        current_ranges = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        
        # 3. Get target ranges from URDF (ground truth size)
        # Assuming obj_size contains [length, diameter, diameter] for a rod
        target_ranges = np.array([1, 1, 1])
        
        # 4. Compute anisotropic scale factors for each axis
        # Avoid division by zero
        scale_factors = target_ranges / current_ranges
        
        # 5. Apply scaling to normalize the shape
        normalized_pc = centered_pc * scale_factors
        normalize_com = (center_of_mass_pcd) * scale_factors
        
        # 6. (Optional) Align with URDF's COM if needed
        # normalized_pc += (center_of_mass_rod - pcd_center)
        
        return normalized_pc, scale_factors, pcd_center, normalize_com
    
    def _get_urdf_shape(self, urdf_path):
        obj_dir = os.path.dirname(pybullet_data.__file__)
        urdf_path = os.path.join(obj_dir, urdf_path)
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('link'):
            visual = link.find('visual')
            if visual is not None:
                geometry = visual.find('geometry')
                if geometry is not None:
                    shape_type = list(geometry)[0].tag
                    if shape_type == 'mesh':
                        filename = geometry.find('mesh').get('filename')
                        scale = geometry.find('mesh').get('scale')
                        # print(f"  Visual Shape: Mesh, Filename: {filename}, Scale: {scale}")

            collision = link.find('collision')
            if collision is not None:
                geometry = collision.find('geometry')
                if geometry is not None:
                    shape_type = list(geometry)[0].tag
                    if shape_type == 'box':
                        size = geometry.find('box').get('size')
                    elif shape_type == 'mesh':
                        size = geometry.find('mesh').get('scale')
                        # print(f"  Collision Shape: Box, Size: {size}")

            inertial = link.find('inertial')
            if inertial is not None:
                # mass = inertial.find('mass').get('value')
                origin = inertial.find('origin')
                if origin is not None:
                    position = origin.get('xyz')
                    # print(f"  Mass: {mass}, Position: {position}")
        
        assert scale == size, "Scale and size should be the same"
        size = [float(x) for x in size.split(' ')]
        position = [float(x) for x in position.split(' ')]
        return size, position
    
    def get_image(self):
        import cv2
        obs = self.camera.render(self.client_id, self.with_plane) # link seg
        rgb = obs["rgb"]
        rgb_img = np.reshape(rgb, (480, 640, 4))
        rgb_img = rgb_img[:, :, :3]
        output_path = "camera_image.png"

        import matplotlib.pyplot as plt
        plt.imshow(rgb_img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()

def visualize_pcd(points, center_of_mass=None):
    import plotly.graph_objects as go
    import plotly.io as pio
    
    pio.renderers.default = "notebook"
    
    figure_data = [
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 2],
                colorscale='Viridis',
                opacity=0.8
            )
        )
    ]
    
    if center_of_mass is not None:
        figure_data.append(
            go.Scatter3d(
                x=[center_of_mass[0]],
                y=[center_of_mass[1]],
                z=[center_of_mass[2]],
                mode='markers',
                marker=dict(
                    size=5,  # 稍微大一点，以便突出显示
                    color='red',  # 设置为红色
                    opacity=1.0
                )
            )
        )
    
    fig = go.Figure(data=figure_data)

    fig.update_layout(
        title='3D Point Cloud Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.write_image('~/uneven-object-go/scripts/debug/debug.jpg')
    
    fig.show()