import copy
import time
from dataclasses import dataclass
from typing import Optional, List

import imageio
import numpy as np
import pybullet as p
import torch
from flowbot3d.datasets.flow_dataset import compute_normalized_flow
from flowbot3d.grasping.agents.flowbot3d import FlowNetAnimation
from rpad.partnet_mobility_utils.data import PMObject
from rpad.partnet_mobility_utils.render.pybullet import PMRenderEnv
from rpad.pybullet_envs.suction_gripper import FloatingSuctionGripper
from scipy.spatial.transform import Rotation as R

from have.generator.datasets.flow_trajectory_dataset import (
    compute_flow_trajectory,
)
from have.generator.metrics.trajectory import normalize_trajectory


class PMSuctionSim:
    def __init__(self, obj_id: str, dataset_path: str, gui: bool = False, camera_pos: List = [-2, 0, 2], camera_pos_random: bool=False, gripper: bool=True):
        self.render_env = PMRenderEnv(obj_id=obj_id, dataset_path=dataset_path, gui=gui, camera_pos=camera_pos)
        if camera_pos_random:
            self.render_env.set_camera(camera_xyz="random")
        self.gui = gui
        p.resetDebugVisualizerCamera(
            cameraTargetPosition=[0, 0, 0],
            cameraDistance=5,
            cameraYaw=220,
            # distance=3,
            # yaw=180,
            cameraPitch=-30,
        )
        if gripper:
            self.gripper = FloatingSuctionGripper(self.render_env.client_id)
            self.gripper.set_pose(
                [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
            )
        self.writer = None

    # def run_demo(self):
    #     while True:
    #         self.gripper.set_velocity([0.4, 0, 0.0], [0, 0, 0])
    #         for i in range(10):
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)
    #         contact = self.gripper.detect_contact()
    #         if contact:
    #             break

    #     print("stopping gripper")

    #     self.gripper.set_velocity([0.001, 0, 0.0], [0, 0, 0])
    #     for i in range(10):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)
    #         contact = self.gripper.detect_contact()
    #         print(contact)

    #     print("starting activation")

    #     self.gripper.activate()

    #     self.gripper.set_velocity([0, 0, 0.0], [0, 0, 0])
    #     for i in range(100):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    #     # print("releasing")
    #     # self.gripper.release()

    #     print("starting motion")
    #     for i in range(100):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    #     for _ in range(20):
    #         for i in range(100):
    #             self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
    #             self.gripper.apply_force([-500, 0, 0])
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)

    #         for i in range(100):
    #             self.gripper.set_velocity([-0.4, 0, 0.0], [0, 0, 0])
    #             self.gripper.apply_force([-500, 0, 0])
    #             p.stepSimulation(self.render_env.client_id)
    #             time.sleep(1 / 240.0)

    #     print("releasing")
    #     self.gripper.release()

    #     for i in range(1000):
    #         p.stepSimulation(self.render_env.client_id)
    #         time.sleep(1 / 240.0)

    def reset(self):
        pass

    def set_writer(self, writer):
        self.writer = writer

    def disable_self_collision(self):
        # Disable the gripper's collision with the floor
        for gripper_part_id in [self.gripper.base_id, self.gripper.body_id]:
            num_links = p.getNumJoints(gripper_part_id)
            for gripper_link_id in range(-1, num_links):
                p.setCollisionFilterPair(
                    bodyUniqueIdA=gripper_part_id, 
                    bodyUniqueIdB=self.render_env.plane_id,
                    linkIndexA=gripper_link_id,
                    linkIndexB=-1,
                    enableCollision=0,
                    physicsClientId=self.render_env.client_id
                )
        # Disable the self collision of the object

        # Disable the self collision of the gripper

    def disable_collision(self, link_id, base=True, body=True, floor=False):
        if base:
            num_links = p.getNumJoints(self.gripper.base_id)
            for gripper_link_id in range(-1, num_links):
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.gripper.base_id, 
                    bodyUniqueIdB=self.render_env.obj_id,
                    linkIndexA=gripper_link_id,
                    linkIndexB=link_id,
                    enableCollision=0,
                    physicsClientId=self.render_env.client_id
                )
            # p.setCollisionFilterPair(
            #     bodyUniqueIdA=self.gripper.base_id, 
            #     bodyUniqueIdB=self.render_env.obj_id,
            #     linkIndexA=0,
            #     linkIndexB=link_id,
            #     enableCollision=0,
            #     physicsClientId=self.render_env.client_id
            # )
        if body:
            num_links = p.getNumJoints(self.gripper.body_id)
            for gripper_link_id in range(-1, num_links):
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.gripper.body_id, 
                    bodyUniqueIdB=self.render_env.obj_id,
                    linkIndexA=gripper_link_id,
                    linkIndexB=link_id,
                    enableCollision=0,
                    physicsClientId=self.render_env.client_id
                )
            # p.setCollisionFilterPair(
            #     bodyUniqueIdA=self.gripper.body_id, 
            #     bodyUniqueIdB=self.render_env.obj_id,
            #     linkIndexA=-1,
            #     linkIndexB=link_id,
            #     enableCollision=0,
            #     physicsClientId=self.render_env.client_id
            # )
            # p.setCollisionFilterPair(
            #     bodyUniqueIdA=self.gripper.body_id, 
            #     bodyUniqueIdB=self.render_env.obj_id,
            #     linkIndexA=0,
            #     linkIndexB=link_id,
            #     enableCollision=0,
            #     physicsClientId=self.render_env.client_id
            # )
        else:  # Always disable the collision with -1 link of the gripper tip
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.gripper.body_id, 
                bodyUniqueIdB=self.render_env.obj_id,
                linkIndexA=-1,
                linkIndexB=link_id,
                enableCollision=0,
                physicsClientId=self.render_env.client_id
            )
        if floor:
            # Disable collision with floor
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.render_env.plane_id, 
                bodyUniqueIdB=self.render_env.obj_id,
                linkIndexA=-1,
                linkIndexB=link_id,
                enableCollision=0,
                physicsClientId=self.render_env.client_id
            )
        p.stepSimulation()

    def reset_gripper(self, target_link):
        # print(self.gripper.contact_const)
        curr_pos = self.get_joint_value(target_link)
        self.gripper.release()
        self.gripper.set_pose(
            [-1, 0.6, 0.8], p.getQuaternionFromEuler([0, np.pi / 2, 0])
        )
        self.gripper.set_velocity([0, 0, 0], [0, 0, 0])
        self.set_joint_state(target_link, curr_pos)

    def set_gripper_pose(self, pos, ori):
        self.gripper.set_pose(pos, ori)

    def set_joint_state(self, link_name: str, value: float):
        p.resetJointState(
            self.render_env.obj_id,
            self.render_env.link_name_to_index[link_name],
            value,
            0.0,
            self.render_env.client_id,
        )

    def render(self, filter_nonobj_pts: bool = False, n_pts: Optional[int] = None, return_pcd_to_pixel=False):
        output = self.render_env.render()
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = output
        pcd_to_pixel = np.arange(rgb.shape[0] * rgb.shape[1])
        pcd_to_pixel = pcd_to_pixel[seg.reshape(-1) > 0]
        # print(len(pcd_to_pixel), rgb.shape, len(P_world))
        assert len(pcd_to_pixel) == len(P_world)

        if filter_nonobj_pts:
            pc_seg_obj = np.ones_like(pc_seg) * -1
            for k, (body, link) in segmap.items():
                if body == self.render_env.obj_id:
                    ixs = pc_seg == k
                    pc_seg_obj[ixs] = link

            is_obj = pc_seg_obj != -1
            pcd_to_pixel = pcd_to_pixel[is_obj]
            P_cam = P_cam[is_obj]
            P_world = P_world[is_obj]
            pc_seg = pc_seg_obj[is_obj]
        if n_pts is not None:
            perm = np.random.permutation(len(P_world))[:n_pts]
            pcd_to_pixel = pcd_to_pixel[perm]
            P_cam = P_cam[perm]
            P_world = P_world[perm]
            pc_seg = pc_seg[perm]

        if return_pcd_to_pixel:
            return rgb, depth, seg, P_cam, P_world, pc_seg, segmap, pcd_to_pixel
        else:
            return rgb, depth, seg, P_cam, P_world, pc_seg, segmap

    def set_camera(self):
        pass

    def teleport_and_approach(
        self, point, contact_vector, video_writer=None, standoff_d: float = 0.2, print_log=True
    ):
        # Normalize contact vector.
        contact_vector = (contact_vector / contact_vector.norm(dim=-1)).float()

        p_teleport = (torch.from_numpy(point) + contact_vector * standoff_d).float()

        # breakpoint()

        e_z_init = torch.tensor([0, 0, 1.0]).float()
        e_y = -contact_vector
        e_x = torch.cross(-contact_vector, e_z_init)
        e_x = e_x / e_x.norm(dim=-1)
        e_z = torch.cross(e_x, e_y)
        e_z = e_z / e_z.norm(dim=-1)
        R_teleport = torch.stack([e_x, e_y, e_z], dim=1)
        R_gripper = torch.as_tensor(
            [
                [1, 0, 0],
                [0, 0, 1.0],
                [0, -1.0, 0],
            ]
        )
        # breakpoint()
        o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()

        self.gripper.set_pose(p_teleport, o_teleport)

        contact = self.gripper.detect_contact(self.render_env.obj_id)
        max_steps = 500
        curr_steps = 0
        self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
        while not contact and curr_steps < max_steps:
            p.stepSimulation(self.render_env.client_id)

            if video_writer is not None and curr_steps % 50 == 49:
                # if video_writer is not None:
                frame_width = 640
                frame_height = 480
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=frame_width,
                    height=frame_height,
                    viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 0, 0],
                        distance=5,
                        yaw=270,
                        # distance=3,
                        # yaw=180,
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
                video_writer.append_data(image)

            curr_steps += 1
            if self.gui:
                time.sleep(1 / 240.0)
            if curr_steps % 1 == 0:
                contact = self.gripper.detect_contact(self.render_env.obj_id)

        # Give it another chance
        if contact:
            if print_log:
                print("contact detected")

        return contact

    def teleport(
        self,
        points,
        contact_vectors,
        video_writer=None,
        standoff_d: float = 0.2,
        target_link=None,
        print_log=True,
        valid_contact_points=None,
    ):
        # p.setTimeStep(1.0/240)
        orig_angle = self.get_joint_value(target_link)
        for id, (point, contact_vector) in enumerate(zip(points, contact_vectors)):
            # Normalize contact vector.
            # print("CONTACT NORM", contact_vector.norm(dim=-1))
            # if contact_vector.norm(dim=-1) < 0.5:  # Filter out meaningless pull point
            #     return -1, False
            contact_vector = (contact_vector / contact_vector.norm(dim=-1)).float()
            p_teleport = (torch.from_numpy(point) + contact_vector * standoff_d).float()
            # print(p_teleport)
            e_z_init = torch.tensor([0, 0, 1.0]).float()
            e_y = -contact_vector
            e_x = torch.cross(-contact_vector, e_z_init)
            e_x = e_x / e_x.norm(dim=-1)
            e_z = torch.cross(e_x, e_y)
            e_z = e_z / e_z.norm(dim=-1)
            R_teleport = torch.stack([e_x, e_y, e_z], dim=1)
            R_gripper = torch.as_tensor(
                [
                    [1, 0, 0],
                    [0, 0, 1.0],
                    [0, -1.0, 0],
                ]
            )
            o_teleport = R.from_matrix(R_teleport @ R_gripper).as_quat()
            self.gripper.set_pose(p_teleport, o_teleport)
            p.stepSimulation(self.render_env.client_id)
            # print("DEBUG: After set pose")
            # breakpoint()

            # print("DEBUG: After set pose")
            # breakpoint()

            contact = self.gripper.detect_contact(self.render_env.obj_id)
            max_steps = 500
            curr_steps = 0
            # self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
            while not contact and curr_steps < max_steps:
                self.gripper.set_velocity(-contact_vector * 0.4, [0, 0, 0])
                p.stepSimulation(self.render_env.client_id)

                # print(f"DEBUG: After contact step {curr_steps}")
                # breakpoint()
                # print(point, p.getBasePositionAndOrientation(self.gripper.body_id),p.getBasePositionAndOrientation(self.gripper.base_id))
                # if video_writer is not None and curr_steps % 50 == 49:
                if video_writer is not None and False:  # Don't save this
                    # if video_writer is not None and True:
                    # if video_writer is not None:
                    # for i in range(10 if curr_steps == max_steps - 1 else 1):
                    frame_width = 640
                    frame_height = 480
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
                    # if curr_steps == 0:
                    #     cv2.imwrite('/home/yishu/failure_recovery/src/failure_recovery/simulations/logs/simu_eval/video_assets/static_frames/reset_gripper.jpg', image)
                    # if curr_steps == max_steps - 1:
                    #     cv2.imwrite('/home/yishu/failure_recovery/src/failure_recovery/simulations/logs/simu_eval/video_assets/static_frames/attach_gripper.jpg', image)
                    # Add the frame to the video
                    video_writer.append_data(image)

                curr_steps += 1
                if self.gui:
                    time.sleep(1 / 240.0)
                if curr_steps % 1 == 0:
                    contact = self.gripper.detect_contact(self.render_env.obj_id)

            
            # Give it another chance
            if contact:
                if print_log:
                    print("contact detected")
                return id, True
            
            if target_link is not None:
                # link_index = self.render_env.link_name_to_index[target_link]
                curr_pos = self.get_joint_value(target_link)
                # print("Current pos:", curr_pos)
                if curr_pos != orig_angle:
                    self.set_joint_state(target_link, orig_angle)
            
        return -1, False

    def attach(self):
        self.gripper.activate(self.render_env.obj_id)

    def pull(self, direction, n_steps: int = 100):
        direction = torch.as_tensor(direction)
        # print("NORM", direction.norm(dim=-1))
        # if direction.norm(dim=-1) < 0.5:  # Filter out meaningless pull point
        #     return False
        direction = direction / direction.norm(dim=-1)
        # breakpoint()
        for _ in range(n_steps):
            self.gripper.set_velocity(direction * 0.4, [0, 0, 0])
            # self.gripper.apply_force(direction * 50)
            p.stepSimulation(self.render_env.client_id)
            if self.gui:
                time.sleep(1 / 240.0)

        return False

    def pull_with_constraint(
        self, direction, n_steps: int = 100, target_link: str = "", constraint=True
    ):
        if not constraint:
            return self.pull(direction, n_steps)
        # Link info
        link_index = self.render_env.link_name_to_index[target_link]
        info = p.getJointInfo(
            self.render_env.obj_id, link_index, self.render_env.client_id
        )
        lower, upper = info[8], info[9]
        
        direction = torch.as_tensor(direction)
        # print("NORM", direction.norm(dim=-1))
        # if direction.norm(dim=-1) < 0.5:  # Filter out meaningless pull point
        #     return False
        direction = direction / (direction.norm(dim=-1) + 1e-12)
        for _ in range(n_steps):
            self.gripper.set_velocity(direction * 0.4, [0, 0, 0])
            p.stepSimulation(self.render_env.client_id)
            
            # print(self.get_joint_value(target_link))

            # print(f"DEBUG: After pull step {_}")
            # breakpoint()

            # frame_width = 640
            # frame_height = 480
            # width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            #     width=frame_width,
            #     height=frame_height,
            #     viewMatrix=p.computeViewMatrixFromYawPitchRoll(
            #         cameraTargetPosition=[0, 0, 0],
            #         distance=5,
            #         yaw=270,
            #         # yaw=90,
            #         pitch=-30,
            #         roll=0,
            #         upAxisIndex=2,
            #     ),
            #     projectionMatrix=p.computeProjectionMatrixFOV(
            #         fov=60,
            #         aspect=float(frame_width) / frame_height,
            #         nearVal=0.1,
            #         farVal=100.0,
            #     ),
            # )
            # image = np.array(rgbImg, dtype=np.uint8)
            # image = image[:, :, :3]
            # # Add the frame to the video
            # self.writer.append_data(image)
            # print("Current !: ", self.get_joint_value(target_link))

            if self.gui:
                time.sleep(1 / 240.0)

        # Check if the object is below initial_angle
        curr_pos = self.get_joint_value(target_link)
        if curr_pos < lower < upper or curr_pos > lower > upper:
            print(curr_pos, lower)
            p.resetJointState(
                self.render_env.obj_id, link_index, lower, 0, self.render_env.client_id
            )
            return True  # Need a reset

        return False  # Don't need reset

    def get_joint_value(self, target_link: str):
        link_index = self.render_env.link_name_to_index[target_link]
        state = p.getJointState(
            self.render_env.obj_id, link_index, self.render_env.client_id
        )
        joint_pos = state[0]
        return joint_pos

    def detect_success(self, target_link: str):
        link_index = self.render_env.link_name_to_index[target_link]
        info = p.getJointInfo(
            self.render_env.obj_id, link_index, self.render_env.client_id
        )
        lower, upper = info[8], info[9]
        curr_pos = self.get_joint_value(target_link)

        sign = -1 if upper < lower else 1
        print(
            f"lower: {lower}, upper: {upper}, curr: {curr_pos}, success:{(upper - curr_pos) / (upper - lower) < 0.1}"
        )

        return (upper - curr_pos) / (upper - lower) < 0.1, (curr_pos - lower) / (
            upper - lower
        )

    def randomize_joints(self):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[2] == p.JOINT_REVOLUTE or jinfo[2] == p.JOINT_PRISMATIC:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def randomize_specific_joints(self, joint_list):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = np.random.random() * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def articulate_specific_joints(self, joint_list, amount):
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angle = amount * (upper - lower) + lower
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )

    def randomize_joints_openclose(self, joint_list):
        randind = np.random.choice([0, 1])
        # Close: 0
        # Open: 1
        self.close_or_open = randind
        for i in range(
            p.getNumJoints(self.render_env.obj_id, self.render_env.client_id)
        ):
            jinfo = p.getJointInfo(self.render_env.obj_id, i, self.render_env.client_id)
            if jinfo[12].decode("UTF-8") in joint_list:
                lower, upper = jinfo[8], jinfo[9]
                angles = [lower, upper]
                angle = angles[randind]
                p.resetJointState(
                    self.render_env.obj_id, i, angle, 0, self.render_env.client_id
                )


@dataclass
class TrialResult:
    success: bool
    contact: bool
    assertion: bool
    init_angle: float
    final_angle: float
    now_angle: float

    # UMPNet metric goes here
    metric: float


class GTFlowModel:
    def __init__(self, raw_data, env):
        self.env = env
        self.raw_data = raw_data

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        links = raw_data.semantics.by_type("slider")
        links += raw_data.semantics.by_type("hinge")
        current_jas = {}
        for link in links:
            linkname = link.name
            chain = raw_data.obj.get_chain(linkname)
            for joint in chain:
                current_jas[joint.name] = 0

        normalized_flow = compute_normalized_flow(
            P_world,
            env.render_env.T_world_base,
            current_jas,
            pc_seg,
            env.render_env.link_name_to_index,
            raw_data,
            "all",
        )

        return torch.from_numpy(normalized_flow)

    def get_movable_mask(self, obs) -> torch.Tensor:
        flow = self(obs)
        mask = (~(np.isclose(flow, 0.0)).all(axis=-1)).astype(np.bool_)
        return mask


class GTTrajectoryModel:
    def __init__(self, raw_data, env, traj_len=20):
        self.raw_data = raw_data
        self.env = env
        self.traj_len = traj_len

    def __call__(self, obs) -> torch.Tensor:
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = obs
        env = self.env
        raw_data = self.raw_data

        links = raw_data.semantics.by_type("slider")
        links += raw_data.semantics.by_type("hinge")
        current_jas = {}
        for link in links:
            linkname = link.name
            chain = raw_data.obj.get_chain(linkname)
            for joint in chain:
                current_jas[joint.name] = 0
        trajectory, _ = compute_flow_trajectory(
            self.traj_len,
            P_world,
            env.render_env.T_world_base,
            current_jas,
            pc_seg,
            env.render_env.link_name_to_index,
            raw_data,
            "all",
        )
        return torch.from_numpy(trajectory)

    def get_gt_force_vector(self, obs, link_ixs) -> torch.Tensor:  # Just for debug!!!!
        pred_flow = self(obs)[link_ixs, 0, :]
        best_flow_ix = torch.topk(pred_flow.norm(dim=-1), 1)[1]
        # breakpoint()
        return pred_flow[best_flow_ix] / pred_flow[best_flow_ix].norm(2)


def choose_grasp_points(
    raw_pred_flow, raw_point_cloud, filter_edge=False, k=40, last_correct_direction=None
):
    pred_flow = raw_pred_flow.clone()
    point_cloud = raw_point_cloud
    # Choose top k non-edge grasp points:
    if filter_edge:  # Need to filter the edge points
        squared_diff = (
            point_cloud[:, np.newaxis, :] - point_cloud[np.newaxis, :, :]
        ) ** 2
        dists = np.sqrt(np.sum(squared_diff, axis=2))
        dist_thres = np.percentile(dists, 10)
        neighbour_points = np.sum(dists < dist_thres, axis=0)
        invalid_points = neighbour_points < np.percentile(
            neighbour_points, 30
        )  # Not edge
        pred_flow[invalid_points] = 0  # Don't choose these edge points!!!!!

    top_k_point = min(k, len(pred_flow))
    best_flow_ix = torch.topk(pred_flow.norm(dim=-1), top_k_point)[1]
    if top_k_point == 1:
        best_flow_ix = torch.tensor(list(best_flow_ix) * 2)
    best_flow = pred_flow[best_flow_ix]
    best_point = point_cloud[best_flow_ix]

    if last_correct_direction is None:  # No past direction as filter
        # print(best_flow_ix.shape, best_flow.shape, best_point.shape)
        return best_flow_ix, best_flow, best_point
    else:
        filtered_best_flow_ix = []
        filtered_best_flow = []
        filtered_best_point = []
        for ix, flow, point in zip(best_flow_ix, best_flow, best_point):
            # if np.dot(flow, last_correct_direction) > 0:  # angle < 90
            if (
                np.dot(
                    flow / (np.linalg.norm(flow) + 1e-12),
                    last_correct_direction
                    / (np.linalg.norm(last_correct_direction) + 1e-12),
                )
                > 0.80
            ):  # angle < 60
                # print("last correct_direction: ", last_correct_direction / np.linalg.norm(last_correct_direction), "Current proposal:", flow / (np.linalg.norm(flow) + 1e-12))
                # print("good prediction:", ix, flow, point, np.dot(flow / np.linalg.norm(flow), last_correct_direction / np.linalg.norm(last_correct_direction)))
                filtered_best_flow_ix.append(ix)
                filtered_best_flow.append(flow)
                filtered_best_point.append(point)
            # else:
            #     print("last correct_direction: ", last_correct_direction / np.linalg.norm(last_correct_direction), "Current proposal:", flow / (np.linalg.norm(flow) + 1e-12))
            #     print("good prediction:", ix, flow, point, np.dot(flow / np.linalg.norm(flow), last_correct_direction / np.linalg.norm(last_correct_direction)))

        if len(filtered_best_flow) == 0:
            return [], [], []
        return (
            torch.stack(filtered_best_flow_ix),
            torch.stack(filtered_best_flow),
            np.array(filtered_best_point),
        )


def choose_grasp_points_density(
    raw_pred_flow, raw_point_cloud, k=40, last_correct_direction=None
):
    pred_flow = raw_pred_flow.clone()
    point_cloud = raw_point_cloud

    flow_norms = pred_flow.norm(dim=-1)
    point_density = flow_norms / flow_norms.sum()
    # breakpoint()
    choices = np.arange(0, len(point_density))

    top_k_point = min(k, len(pred_flow))
    # best_flow_ix = torch.topk(pred_flow.norm(dim=-1), top_k_point)[1]
    best_flow_ix = np.random.choice(
        choices, size=top_k_point, replace=False, p=point_density.numpy()
    )
    best_flow_ix = torch.from_numpy(best_flow_ix)
    if top_k_point == 1:
        best_flow_ix = torch.tensor(list(best_flow_ix) * 2)
    best_flow = pred_flow[best_flow_ix]
    best_point = point_cloud[best_flow_ix]

    if last_correct_direction is None:  # No past direction as filter
        # print(best_flow_ix.shape, best_flow.shape, best_point.shape)
        return best_flow_ix, best_flow, best_point
    else:
        filtered_best_flow_ix = []
        filtered_best_flow = []
        filtered_best_point = []
        for ix, flow, point in zip(best_flow_ix, best_flow, best_point):
            # if np.dot(flow, last_correct_direction) > 0:  # angle < 90
            if (
                np.dot(
                    flow / (np.linalg.norm(flow) + 1e-12),
                    last_correct_direction
                    / (np.linalg.norm(last_correct_direction) + 1e-12),
                )
                > 0.80
            ):  # angle < 60
                # print("last correct_direction: ", last_correct_direction / np.linalg.norm(last_correct_direction))
                # print("good prediction:", ix, flow, point, np.dot(flow / np.linalg.norm(flow), last_correct_direction / np.linalg.norm(last_correct_direction)))
                filtered_best_flow_ix.append(ix)
                filtered_best_flow.append(flow)
                filtered_best_point.append(point)

        if len(filtered_best_flow) == 0:
            return [], [], []
        return (
            torch.stack(filtered_best_flow_ix),
            torch.stack(filtered_best_flow),
            np.array(filtered_best_point),
        )


def point_direction_to_grasp_field(P_world, link_ixs, best_point, best_flow, grasp_selection=False, normalize=False):
    if torch.is_tensor(best_flow):
        best_flow_np = best_flow.numpy()
    else:
        best_flow_np = best_flow
    if normalize:
        best_flow_np = best_flow_np / np.linalg.norm(best_flow_np)
    grasp_flow_field = np.zeros_like(P_world) + best_flow_np[np.newaxis, :]
    pcd_dist = np.power(P_world - best_point, 2).sum(axis=-1)
    if grasp_selection:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
    else:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
    grasp_flow_field = grasp_flow_field * coeff
    grasp_flow_field *= link_ixs[:, np.newaxis]
    return grasp_flow_field


def flow_to_grasp_field(pred_flow, P_world, link_ixs=None, grasp_selection=False):
    norms = np.linalg.norm(pred_flow[link_ixs], axis=-1)
    grasp_point_idx = np.argmax(norms)
    best_flow = pred_flow[link_ixs][grasp_point_idx].numpy()
    best_point = P_world[link_ixs][grasp_point_idx]
    best_flow /= np.linalg.norm(best_flow)
    grasp_flow_field = np.zeros_like(P_world) + best_flow[np.newaxis, :]
    pcd_dist = np.power(P_world - best_point, 2).sum(axis=-1)
    if grasp_selection:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis]
    else:
        coeff = np.exp(-5 * pcd_dist)[:, np.newaxis] 
    grasp_flow_field = grasp_flow_field * coeff
    if link_ixs is not None:
        grasp_flow_field *= link_ixs[:, np.newaxis]
    return grasp_flow_field