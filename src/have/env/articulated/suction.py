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


def get_local_point(object_id, link_index, world_point):
    if link_index == -1:
        # Base link (root link)
        position, orientation = p.getBasePositionAndOrientation(object_id)
    else:
        # Specific link
        link_state = p.getLinkState(object_id, link_index)
        position = link_state[4]  # Link world position
        orientation = link_state[5]  # Link world orientation

    # Convert orientation to a rotation matrix
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

    # Transform the world point to local coordinates
    local_point = np.dot(
        np.linalg.inv(rotation_matrix), (world_point - np.array(position))
    )
    return local_point


def get_world_point(object_id, link_index, local_point):
    if link_index == -1:
        # Base link (root link)
        position, orientation = p.getBasePositionAndOrientation(object_id)
    else:
        # Specific link
        link_state = p.getLinkState(object_id, link_index)
        position = link_state[4]  # Link world position
        orientation = link_state[5]  # Link world orientation

    # Convert orientation to a rotation matrix
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

    # Transform the local point to world coordinates
    world_point = np.dot(rotation_matrix, local_point) + np.array(position)
    return world_point


def run_trial(
    env: PMSuctionSim,
    raw_data: PMObject,
    target_link: str,
    model,
    gt_model=None,  # When we use mask_input_channel=True, this is the mask generator
    n_steps: int = 30,
    n_pts: int = 1200,
    save_name: str = "unknown",
    website: bool = False,
    gui: bool = False,
    sgp: bool = True,
    consistency_check: bool = False,
    analysis: bool = False,
) -> TrialResult:
    torch.manual_seed(42)
    torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    np.set_printoptions(precision=10)
    # p.setPhysicsEngineParameter(numSolverIterations=10)
    # p.setPhysicsEngineParameter(contactBreakingThreshold=0.01, contactSlop=0.001)

    initial_movement_thres = 1e-6
    good_movement_thres = 0.01
    max_trial_per_step = 50
    this_step_trial = 0

    sim_trajectory = [0.0] + [0] * (n_steps)  # start from 0.05
    correct_direction_stack = []  # The direction stack

    # For analysis:
    sgp_signals = [1]  # Record the steps which we switched grasp point

    # For website demo
    if analysis:
        visual_all_points = []
        visual_link_ixs = []
        visual_grasp_points_idx = []
        visual_grasp_points = []
        visual_flows = []

    if website:
        # Flow animation
        animation = FlowNetAnimation()

    # First, reset the environment.
    env.reset()
    # Joint information
    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    # Sometimes doors collide with themselves. It's dumb.
    if (
        raw_data.category == "Door"
        and raw_data.semantics.by_name(target_link).type == "hinge"
    ):
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.05)

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

    if init_angle == target_angle:  # Not movable
        p.disconnect(physicsClientId=env.render_env.client_id)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    # breakpoint()
    if gt_model is None:  # GT Flow model
        pred_trajectory = model(copy.deepcopy(pc_obs))
    else:
        movable_mask = gt_model.get_movable_mask(pc_obs)
        pred_trajectory = model(copy.deepcopy(pc_obs), movable_mask)
    # pred_trajectory = model(copy.deepcopy(pc_obs))
    # breakpoint()
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    traj_len = pred_trajectory.shape[1]  # Trajectory length
    print(f"Predicting {traj_len} length trajectories.")
    pred_flow = pred_trajectory[:, 0, :]

    # flow_fig(torch.from_numpy(P_world), pred_flow, sizeref=0.1, use_v2=True).show()
    # breakpoint()

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
    # assert link_ixs.any()
    if not link_ixs.any():
        p.disconnect(physicsClientId=env.render_env.client_id)
        print("link_ixs finds no point")
        animation_results = animation.animate() if website else None
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    if website:
        if gui:
            # Record simulation video
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                f"./logs/simu_eval/video_assets/{save_name}.mp4",
            )
        else:
            video_file = f"./logs/simu_eval/video_assets/{save_name}.mp4"
            # # cv2 output videos won't show on website
            frame_width = 640
            frame_height = 480
            # fps = 5
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
            # videoWriter.write(rgbImgOpenCV)

            # Camera param
            writer = imageio.get_writer(video_file, fps=5)

            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=frame_width,
                height=frame_height,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=5,
                    # yaw=180,
                    yaw=270,
                    # pitch=90,
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
            writer.append_data(image)

    # The attachment point is the point with the highest flow.
    # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
    best_flow_ixs, best_flows, best_points = choose_grasp_points(
        pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=20
    )

    # Teleport to an approach pose, approach, the object and grasp.
    if website and not gui:
        # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
        best_flow_ix_id, contact = env.teleport(
            best_points, best_flows, video_writer=writer
        )
    else:
        # contact = env.teleport_and_approach(best_point, best_flow)
        best_flow_ix_id, contact = env.teleport(best_points, best_flows)
    best_flow = pred_flow[link_ixs][best_flow_ixs[best_flow_ix_id]]
    best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]
    last_step_grasp_point = best_point
    # For website demo
    if analysis:
        visual_all_points.append(P_world)
        visual_link_ixs.append(link_ixs)
        visual_grasp_points_idx.append(best_flow_ixs[best_flow_ix_id])
        visual_grasp_points.append(best_point)
        visual_flows.append(best_flow)

    if website:
        segmented_flow = np.zeros_like(pred_flow)
        segmented_flow[link_ixs] = pred_flow[link_ixs]
        segmented_flow = np.array(
            normalize_trajectory(
                torch.from_numpy(np.expand_dims(segmented_flow, 1))
            ).squeeze()
        )
        animation.add_trace(
            torch.as_tensor(P_world),
            torch.as_tensor([P_world]),
            torch.as_tensor([segmented_flow * 3]),
            "red",
        )

    if not contact:
        if website:
            if gui:
                p.stopStateLogging(log_id)
            else:
                # Write video
                writer.close()
                # videoWriter.release()

        print("No contact!")
        p.disconnect(physicsClientId=env.render_env.client_id)
        animation_results = None if not website else animation.animate()
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=True,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    env.attach()
    gripper_tip_pos_before = best_point
    gripper_object_contact_local = get_local_point(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        gripper_tip_pos_before,
    )
    reset = env.pull_with_constraint(best_flow, target_link=target_link, constraint=sgp)
    if not reset:
        env.attach()
        gripper_tip_pos_after = get_world_point(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[target_link],
            gripper_object_contact_local,
        )

        delta_gripper = np.array(gripper_tip_pos_after) - np.array(
            gripper_tip_pos_before
        )

        if np.linalg.norm(delta_gripper) > initial_movement_thres:  # Because
            correct_direction_stack.append(delta_gripper)

        last_step_grasp_point = best_point
    else:
        last_step_grasp_point = None

    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    success, sim_trajectory[1] = env.detect_success(target_link=target_link)

    global_step = 1
    # for i in range(n_steps):
    while not success and global_step < n_steps:
        # Predict the flow on the observation.
        if gt_model is None:  # GT Flow model
            pred_trajectory = model(copy.deepcopy(pc_obs))
        else:
            movable_mask = gt_model.get_movable_mask(pc_obs)
            # breakpoint()
            pred_trajectory = model(pc_obs, movable_mask)
            # pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(pred_trajectory.shape[1]):
            if global_step == n_steps:
                break
            global_step += 1
            pred_flow = pred_trajectory[:, traj_step, :]
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

            # Filter down just the points on the target link.
            # breakpoint()
            link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
            # assert link_ixs.any()
            if not link_ixs.any():
                if website:
                    if gui:
                        p.stopStateLogging(log_id)
                    else:
                        writer.close()
                        # videoWriter.release()
                p.disconnect(physicsClientId=env.render_env.client_id)
                print("link_ixs finds no point")
                animation_results = animation.animate() if website else None
                return (
                    animation_results,
                    TrialResult(
                        assertion=False,
                        success=False,
                        contact=False,
                        init_angle=0,
                        final_angle=0,
                        now_angle=0,
                        metric=0,
                    ),
                    sim_trajectory,
                )

            # Get the best direction.
            # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
            best_flow_ixs, best_flows, best_points = choose_grasp_points(
                pred_flow[link_ixs],
                P_world[link_ixs],
                filter_edge=False,
                k=20,
                last_correct_direction=None
                if len(correct_direction_stack) == 0 or not consistency_check
                else correct_direction_stack[-1],
            )

            have_to_execute_incorrect = False

            if (
                len(best_flows) == 0
            ):  # All top 20 points are filtered out! - Not a good prediction - move on!
                this_step_trial += 1
                if (
                    this_step_trial > max_trial_per_step
                ):  # To make the process go on, must make an action!
                    have_to_execute_incorrect = True
                    print("has to execute incorrect!!!")

                    # Density choosing
                    (
                        best_flow_ixs,
                        best_flows,
                        best_points,
                    ) = choose_grasp_points_density(
                        pred_flow[link_ixs],
                        P_world[link_ixs],
                        k=20,
                        last_correct_direction=None,
                    )
                else:
                    continue

            # (1) Strategy 1 - Don't change grasp point
            # # (2) Strategy 2 - Change grasp point when leverage difference is large
            if sgp:
                lev_diff_thres = 0.2
                no_movement_thres = -1
            else:
                # Don't use this policy
                lev_diff_thres = 100
                no_movement_thres = -1
                good_movement_thres = 1000

            # print(f"Trial {this_step_trial} times")
            if last_step_grasp_point is not None:
                gripper_tip_pos, _ = p.getBasePositionAndOrientation(
                    env.gripper.body_id
                )
                pcd_dist = torch.tensor(
                    P_world[link_ixs] - np.array(gripper_tip_pos)
                ).norm(dim=-1)
                grasp_point_id = pcd_dist.argmin()
                lev_diff = best_flows.norm(dim=-1) - pred_flow[link_ixs][
                    grasp_point_id
                ].norm(dim=-1)

            # gripper_movement = torch.from_numpy(
            #     P_world[link_ixs][grasp_point_id] - last_step_grasp_point
            # ).norm()
            # print("gripper: ",gripper_movement)
            # breakpoint()
            # if (
            #     gripper_movement < no_movement_thres or lev_diff[0] > lev_diff_thres
            # ):  # pcd_dist < 0.05 -> didn't move much....
            if last_step_grasp_point is None or lev_diff[0] > lev_diff_thres:
                sgp_signals.append(1)
                env.reset_gripper(target_link)
                p.stepSimulation(
                    env.render_env.client_id
                )  # Make sure the constraint is lifted

                if website and not gui:
                    # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
                    best_flow_ix_id, contact = env.teleport(
                        best_points,
                        best_flows,
                        video_writer=writer,
                        target_link=target_link,
                    )
                else:
                    # contact = env.teleport_and_approach(best_point, best_flow)
                    best_flow_ix_id, contact = env.teleport(
                        best_points, best_flows, target_link=target_link
                    )

                # image.save('/home/yishu/failure_recovery/src/failure_recovery/simulations/logs/simu_eval/video_assets/static_frames/attach_gripper.jpg')
                best_flow = pred_flow[link_ixs][best_flow_ixs[best_flow_ix_id]]
                best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]
                last_step_grasp_point = best_point  # Grasp a new point
                # print("new!", last_step_grasp_point)

                # For website demo
                if analysis:
                    visual_all_points.append(P_world)
                    visual_link_ixs.append(link_ixs)
                    visual_grasp_points_idx.append(best_flow_ixs[best_flow_ix_id])
                    visual_grasp_points.append(best_point)
                    visual_flows.append(best_flow)

                if not contact:
                    if website:
                        segmented_flow = np.zeros_like(pred_flow)
                        segmented_flow[link_ixs] = pred_flow[link_ixs]
                        segmented_flow = np.array(
                            normalize_trajectory(
                                torch.from_numpy(np.expand_dims(segmented_flow, 1))
                            ).squeeze()
                        )
                        animation.add_trace(
                            torch.as_tensor(P_world),
                            torch.as_tensor([P_world]),
                            torch.as_tensor([segmented_flow * 3]),
                            "red",
                        )
                        if gui:
                            p.stopStateLogging(log_id)
                        else:
                            # Write video
                            writer.close()
                            # videoWriter.release()

                    print("No contact!")
                    p.disconnect(physicsClientId=env.render_env.client_id)
                    animation_results = None if not website else animation.animate()
                    return (
                        animation_results,
                        TrialResult(
                            success=False,
                            assertion=True,
                            contact=False,
                            init_angle=0,
                            final_angle=0,
                            now_angle=0,
                            metric=0,
                        ),
                        sim_trajectory,
                    )

                env.attach()
            else:
                sgp_signals.append(0)
                best_flow = pred_flow[link_ixs][best_flow_ixs[0]]
                best_point = P_world[link_ixs][grasp_point_id]
                last_step_grasp_point = best_point
                # The original point - don't need to change
                # print("same:", last_step_grasp_point)

                # For website demo
                if analysis:
                    visual_all_points.append(P_world)
                    visual_link_ixs.append(link_ixs)
                    visual_grasp_points_idx.append(grasp_point_id)
                    visual_grasp_points.append(best_point)
                    visual_flows.append(best_flow)

            # Execute the step:
            env.attach()
            gripper_tip_pos_before = last_step_grasp_point
            gripper_object_contact_local = get_local_point(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[target_link],
                gripper_tip_pos_before,
            )
            reset = env.pull_with_constraint(
                best_flow, target_link=target_link, constraint=sgp
            )
            if not reset:
                env.attach()
                last_step_grasp_point = best_point
                gripper_tip_pos_after = get_world_point(
                    env.render_env.obj_id,
                    env.render_env.link_name_to_index[target_link],
                    gripper_object_contact_local,
                )

                # Now with filter: we guarantee that every step is correct!!
                delta_gripper = np.array(gripper_tip_pos_after) - np.array(
                    gripper_tip_pos_before
                )

                # -----------Update the direction and history stack!!!!-----------
                if len(correct_direction_stack) == 0:
                    # Update direction stack
                    if np.linalg.norm(delta_gripper) > initial_movement_thres:
                        correct_direction_stack.append(
                            delta_gripper / (np.linalg.norm(delta_gripper) + 1e-12)
                        )
                else:
                    # Update direction stack:
                    if (
                        np.dot(delta_gripper, correct_direction_stack[-1]) > 0
                    ):  # Consistent
                        correct_direction_stack.append(
                            delta_gripper / (np.linalg.norm(delta_gripper) + 1e-12)
                        )

            else:  # Need to reset gripper
                last_step_grasp_point = None
            # print(best_flow)
            env.attach()
            # print("After pulling!!", env.get_joint_value(target_link))
            # breakpoint()

            if website:
                # Add pcd to flow animation
                segmented_flow = np.zeros_like(pred_flow)
                segmented_flow[link_ixs] = pred_flow[link_ixs]
                segmented_flow = np.array(
                    normalize_trajectory(
                        torch.from_numpy(np.expand_dims(segmented_flow, 1))
                    ).squeeze()
                )
                animation.add_trace(
                    torch.as_tensor(P_world),
                    torch.as_tensor([P_world]),
                    torch.as_tensor([segmented_flow * 3]),
                    "red",
                )

                # Capture frame
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
                # rgbImgOpenCV = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
                # videoWriter.write(rgbImgOpenCV)
                image = np.array(rgbImg, dtype=np.uint8)
                image = image[:, :, :3]

                # Add the frame to the video
                writer.append_data(image)

            success, sim_trajectory[global_step] = env.detect_success(target_link)

            if success:
                for left_step in range(global_step, 31):
                    sim_trajectory[left_step] = sim_trajectory[global_step]
                break

            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)
            this_step_trial = 0  # This step is executed!

        if success:
            for left_step in range(global_step, 31):
                sim_trajectory[left_step] = sim_trajectory[global_step]
            break

    # calculate the metrics
    curr_pos = env.get_joint_value(target_link)
    metric = (curr_pos - init_angle) / (target_angle - init_angle)
    metric = min(max(metric, 0), 1)

    if website:
        if gui:
            p.stopStateLogging(log_id)
        else:
            writer.close()
            # videoWriter.release()

    p.disconnect(physicsClientId=env.render_env.client_id)
    animation_results = None if not website else animation.animate()
    return (
        animation_results,
        TrialResult(  # Save the flow visuals
            success=success,
            contact=True,
            assertion=True,
            init_angle=init_angle,
            final_angle=target_angle,
            now_angle=curr_pos,
            metric=metric,
        ),
        sim_trajectory
        if not analysis
        else [
            sim_trajectory,
            None,
            None,
            sgp_signals,
            visual_all_points,
            visual_link_ixs,
            visual_grasp_points_idx,
            visual_grasp_points,
            visual_flows,
        ],
    )


# Policy to filter the inconsistent actions and incorrect histories
def run_trial_with_history_filter(
    env: PMSuctionSim,
    raw_data: PMObject,
    target_link: str,
    model,
    model_with_history,
    gt_model=None,  # When we use mask_input_channel=True, this is the mask generator
    n_steps: int = 30,
    n_pts: int = 1200,
    save_name: str = "unknown",
    website: bool = False,
    gui: bool = False,
    consistency_check=True,
    history_filter=True,
    analysis=False,
) -> TrialResult:
    # torch.manual_seed(42)
    torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    np.set_printoptions(precision=10)
    # p.setPhysicsEngineParameter(numSolverIterations=10)
    # p.setPhysicsEngineParameter(contactBreakingThreshold=0.01, contactSlop=0.001)
    print("Use consistency check:", consistency_check)
    print("Use history filter:", history_filter)

    initial_movement_thres = 1e-6
    good_movement_thres = 0.01
    max_trial_per_step = 50
    this_step_trial = 0
    prev_flow_pred = None
    prev_point_cloud = None

    sim_trajectory = [0.0] + [0] * (n_steps)  # start from 0.05
    correct_direction_stack = []  # The direction stack

    # For analysis:
    update_history_step = []  # Record the steps which we update the history
    cc_cnts = []  # Record the consistency failure times we had for each step
    sgp_signals = [1]  # Record the steps which we switched grasp point

    # For website demo
    if analysis:
        visual_all_points = []
        visual_link_ixs = []
        visual_grasp_points_idx = []
        visual_grasp_points = []
        visual_flows = []

    if website:
        # Flow animation
        animation = FlowNetAnimation()

    # First, reset the environment.
    env.reset()
    # Joint information
    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    if (
        raw_data.category == "Door"
        and raw_data.semantics.by_name(target_link).type == "hinge"
    ):
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.05)

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

    if init_angle == target_angle:  # Not movable
        p.disconnect(physicsClientId=env.render_env.client_id)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    # breakpoint()
    if gt_model is None:  # GT Flow model
        pred_trajectory = model(copy.deepcopy(pc_obs))
    else:
        movable_mask = gt_model.get_movable_mask(pc_obs)
        pred_trajectory = model(copy.deepcopy(pc_obs), movable_mask)
    # pred_trajectory = model(copy.deepcopy(pc_obs))
    # breakpoint()
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    traj_len = pred_trajectory.shape[1]  # Trajectory length
    print(f"Predicting {traj_len} length trajectories.")
    pred_flow = pred_trajectory[:, 0, :]

    # flow_fig(torch.from_numpy(P_world), pred_flow, sizeref=0.1, use_v2=True).show()
    # breakpoint()

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
    # assert link_ixs.any()
    if not link_ixs.any():
        p.disconnect(physicsClientId=env.render_env.client_id)
        print("link_ixs finds no point")
        animation_results = animation.animate() if website else None
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    if website:
        if gui:
            # Record simulation video
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                f"./logs/simu_eval/video_assets/{save_name}.mp4",
            )
        else:
            video_file = f"./logs/simu_eval/video_assets/{save_name}.mp4"
            # # cv2 output videos won't show on website
            frame_width = 640
            frame_height = 480
            # fps = 5
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
            # videoWriter.write(rgbImgOpenCV)

            # Camera param
            writer = imageio.get_writer(video_file, fps=5)
            env.set_writer(writer)

            # Capture image
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

            # Add the frame to the video
            writer.append_data(image)

    # The attachment point is the point with the highest flow.
    # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()

    # best_flow_ixs, best_flows, best_points = choose_grasp_points(
    #     pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=40
    # )

    # Density choosing
    best_flow_ixs, best_flows, best_points = choose_grasp_points_density(
        pred_flow[link_ixs], P_world[link_ixs], k=40
    )
    cc_cnts.append(0)

    # Teleport to an approach pose, approach, the object and grasp.
    if website and not gui:
        # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
        best_flow_ix_id, contact = env.teleport(
            best_points, best_flows, video_writer=writer, target_link=target_link
        )
    else:
        # contact = env.teleport_and_approach(best_point, best_flow)
        best_flow_ix_id, contact = env.teleport(
            best_points, best_flows, target_link=target_link
        )

    best_flow = pred_flow[link_ixs][best_flow_ixs[best_flow_ix_id]]
    best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]

    # For website demo
    if analysis:
        visual_all_points.append(P_world)
        visual_link_ixs.append(link_ixs)
        visual_grasp_points_idx.append(best_flow_ixs[best_flow_ix_id])
        visual_grasp_points.append(best_point)
        visual_flows.append(best_flow)

    if website:
        segmented_flow = np.zeros_like(pred_flow)
        segmented_flow[link_ixs] = pred_flow[link_ixs]
        segmented_flow = np.array(
            normalize_trajectory(
                torch.from_numpy(np.expand_dims(segmented_flow, 1))
            ).squeeze()
        )
        animation.add_trace(
            torch.as_tensor(P_world),
            torch.as_tensor([P_world]),
            torch.as_tensor([segmented_flow * 3]),
            "red",
        )

    if not contact:
        if website:
            if gui:
                p.stopStateLogging(log_id)
            else:
                # Write video
                writer.close()
                # videoWriter.release()

        print("No contact!")
        p.disconnect(physicsClientId=env.render_env.client_id)
        animation_results = None if not website else animation.animate()
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=True,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory,
        )

    env.attach()
    use_history = False
    # gripper_tip_pos_before, _ = p.getBasePositionAndOrientation(env.gripper.base_id)
    # points = p.getContactPoints(bodyA=env.gripper.body_id, bodyB=env.render_env.obj_id, linkIndexA=0)
    # assert len(points)!=0, "Contact is None!!!!"
    # gripper_tip_pos_before, _ = points[0][5], points[0][6]
    gripper_tip_pos_before = best_point
    gripper_object_contact_local = get_local_point(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        gripper_tip_pos_before,
    )
    # print(gripper_tip_pos_before, gripper_object_contact_local, get_world_point(env.render_env.obj_id, env.render_env.link_name_to_index[target_link], gripper_object_contact_local))
    # env.pull(best_flow)
    reset = env.pull_with_constraint(best_flow, target_link=target_link)
    if not reset:
        env.attach()
        gripper_tip_pos_after = get_world_point(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[target_link],
            gripper_object_contact_local,
        )

        delta_gripper = np.array(gripper_tip_pos_after) - np.array(
            gripper_tip_pos_before
        )

        last_step_grasp_point = best_point

        if (
            np.linalg.norm(delta_gripper) > initial_movement_thres
        ):  # just to make sure it's not 0, 0, 0
            correct_direction_stack.append(delta_gripper)
        # Judge whether the movement is good - if it's good, update the history! (If not using history_filter, just update the history at every step!)
        if not history_filter or np.linalg.norm(delta_gripper) > good_movement_thres:
            use_history = True
            prev_flow_pred = pred_flow.clone()  # History flow
            prev_point_cloud = copy.deepcopy(P_world)  # History point cloud
            update_history_step.append(1)

    else:  # Need a reset because hit the lower boundary - definitely not a good step
        if history_filter:
            use_history = False
            update_history_step.append(0)
        else:  # no history filter: always update history
            use_history = True
            prev_flow_pred = pred_flow.clone()  # History flow
            prev_point_cloud = copy.deepcopy(P_world)  # History point cloud
            update_history_step.append(1)
        last_step_grasp_point = None  # No contact anymore

    # breakpoint()
    global_step = 1
    success, sim_trajectory[global_step] = env.detect_success(target_link)
    # for i in range(n_steps):
    while not success and global_step < n_steps:
        pc_obs = env.render(
            filter_nonobj_pts=True, n_pts=n_pts
        )  # Render a new point cloud!  #
        # Predict the flow on the observation.
        if gt_model is None:  # GT Flow model
            if use_history:
                print("Using history!")
                # Use history model
                pred_trajectory = model_with_history(
                    copy.deepcopy(pc_obs),
                    copy.deepcopy(prev_point_cloud),
                    copy.deepcopy(prev_flow_pred.numpy()),
                )
            else:
                pred_trajectory = model(copy.deepcopy(pc_obs))
        else:
            movable_mask = gt_model.get_movable_mask(pc_obs)
            # breakpoint()
            pred_trajectory = model(pc_obs, movable_mask)
            # pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        pred_flow = pred_trajectory[:, 0, :]
        rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

        # Filter down just the points on the target link.
        # breakpoint()
        link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
        # assert link_ixs.any()
        if not link_ixs.any():
            if website:
                if gui:
                    p.stopStateLogging(log_id)
                else:
                    writer.close()
                    # videoWriter.release()
            p.disconnect(physicsClientId=env.render_env.client_id)
            print("link_ixs finds no point")
            animation_results = animation.animate() if website else None
            return (
                animation_results,
                TrialResult(
                    assertion=False,
                    success=False,
                    contact=False,
                    init_angle=0,
                    final_angle=0,
                    now_angle=0,
                    metric=0,
                ),
                sim_trajectory,
            )

        # Get the best direction.
        # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
        # ------------DEBUG-------------
        gt_model_debug = GTTrajectoryModel(raw_data, env, 1)
        gt_flow = gt_model_debug.get_gt_force_vector(pc_obs, link_ixs)
        if len(correct_direction_stack) != 0:
            print(
                "GT flow's cosine with the last consistent vector!!!!!",
                np.dot(
                    gt_flow.numpy(),
                    correct_direction_stack[-1]
                    / (np.linalg.norm(correct_direction_stack[-1]) + 1e-12),
                ),
            )
        # ------------DEBUG-------------

        # best_flow_ixs, best_flows, best_points = choose_grasp_points(
        #     pred_flow[link_ixs],
        #     P_world[link_ixs],
        #     filter_edge=False,
        #     k=40,
        #     last_correct_direction=None
        #     if len(correct_direction_stack) == 0
        #     else correct_direction_stack[-1],
        # )

        # Density choosing
        best_flow_ixs, best_flows, best_points = choose_grasp_points_density(
            pred_flow[link_ixs],
            P_world[link_ixs],
            k=20,
            last_correct_direction=None
            if len(correct_direction_stack) == 0 or not consistency_check
            else correct_direction_stack[-1],
        )

        have_to_execute_incorrect = False

        if (
            len(best_flows) == 0
        ):  # All top 20 points are filtered out! - Not a good prediction - move on!
            this_step_trial += 1
            if (
                this_step_trial > max_trial_per_step
            ):  # To make the process go on, must make an action!
                have_to_execute_incorrect = True
                print("has to execute incorrect!!!")
                # best_flow_ixs, best_flows, best_points = choose_grasp_points(
                #     pred_flow[link_ixs],
                #     P_world[link_ixs],
                #     filter_edge=False,
                #     k=20,
                #     last_correct_direction=None,
                # )

                # Density choosing
                best_flow_ixs, best_flows, best_points = choose_grasp_points_density(
                    pred_flow[link_ixs],
                    P_world[link_ixs],
                    k=20,
                    last_correct_direction=None,
                )
            else:
                continue

        cc_cnts.append(this_step_trial)

        # (1) Strategy 1 - Don't change grasp point
        # (2) Strategy 2 - Change grasp point when leverage difference is large
        lev_diff_thres = 0.2
        no_movement_thres = -1

        # # Don't switch grasp point
        # lev_diff_thres = 100
        # no_movement_thres = -1
        # good_movement_thres = 1000

        if last_step_grasp_point is not None:  # Still grasping!
            gripper_tip_pos, _ = p.getBasePositionAndOrientation(env.gripper.body_id)
            pcd_dist = torch.tensor(P_world[link_ixs] - np.array(gripper_tip_pos)).norm(
                dim=-1
            )
            grasp_point_id = pcd_dist.argmin()
            print(grasp_point_id)
            lev_diff = best_flows.norm(dim=-1) - pred_flow[link_ixs][
                grasp_point_id
            ].norm(dim=-1)

        if (  # need to switch grasp point
            last_step_grasp_point is None or lev_diff[0] > lev_diff_thres
        ):
            sgp_signals.append(1)
            env.reset_gripper(target_link)
            p.stepSimulation(
                env.render_env.client_id
            )  # Make sure the constraint is lifted

            if website and not gui:
                # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
                best_flow_ix_id, contact = env.teleport(
                    best_points,
                    best_flows,
                    video_writer=writer,
                    target_link=target_link,
                )
            else:
                # contact = env.teleport_and_approach(best_point, best_flow)
                best_flow_ix_id, contact = env.teleport(
                    best_points, best_flows, target_link=target_link
                )
            best_flow = pred_flow[link_ixs][best_flow_ixs[best_flow_ix_id]]
            best_point = P_world[link_ixs][best_flow_ixs[best_flow_ix_id]]
            last_step_grasp_point = best_point  # Grasp a new point

            # For website demo
            if analysis:
                visual_all_points.append(P_world)
                visual_link_ixs.append(link_ixs)
                visual_grasp_points_idx.append(best_flow_ixs[best_flow_ix_id])
                visual_grasp_points.append(best_point)
                visual_flows.append(best_flow)

            if not contact:
                if website:
                    segmented_flow = np.zeros_like(pred_flow)
                    segmented_flow[link_ixs] = pred_flow[link_ixs]
                    segmented_flow = np.array(
                        normalize_trajectory(
                            torch.from_numpy(np.expand_dims(segmented_flow, 1))
                        ).squeeze()
                    )
                    animation.add_trace(
                        torch.as_tensor(P_world),
                        torch.as_tensor([P_world]),
                        torch.as_tensor([segmented_flow * 3]),
                        "red",
                    )
                    if gui:
                        p.stopStateLogging(log_id)
                    else:
                        # Write video
                        writer.close()
                        # videoWriter.release()

                print("No contact!")
                p.disconnect(physicsClientId=env.render_env.client_id)
                animation_results = None if not website else animation.animate()
                return (
                    animation_results,
                    TrialResult(
                        success=False,
                        assertion=True,
                        contact=False,
                        init_angle=0,
                        final_angle=0,
                        now_angle=0,
                        metric=0,
                    ),
                    sim_trajectory,
                )

            env.attach()
        else:  # Stick to the old grasp point
            sgp_signals.append(0)
            best_flow = pred_flow[link_ixs][best_flow_ixs[0]]
            best_point = P_world[link_ixs][grasp_point_id]
            last_step_grasp_point = (
                best_point  # The original point - don't need to change
            )
            # print("same:", last_step_grasp_point)

            # For website demo
            if analysis:
                visual_all_points.append(P_world)
                visual_link_ixs.append(link_ixs)
                visual_grasp_points_idx.append(grasp_point_id)
                visual_grasp_points.append(best_point)
                visual_flows.append(best_flow)

        # Execute the step:
        print(
            "GT flow's cosine with the predicted vector!!!!!",
            gt_flow,
            best_flow / (np.linalg.norm(best_flow) + 1e-12),
            np.dot(gt_flow.numpy(), best_flow / (np.linalg.norm(best_flow) + 1e-12)),
        )
        env.attach()
        # gripper_tip_pos_before, _ = p.getBasePositionAndOrientation(env.gripper.base_id)
        gripper_tip_pos_before = last_step_grasp_point
        gripper_object_contact_local = get_local_point(
            env.render_env.obj_id,
            env.render_env.link_name_to_index[target_link],
            gripper_tip_pos_before,
        )
        reset = env.pull_with_constraint(best_flow, target_link=target_link)
        # -------DEBUG-------
        # print(gt_flow)
        # reset = env.pull_with_constraint(gt_flow, target_link=target_link)
        # -------DEBUG-------
        if not reset:
            env.attach()
            gripper_tip_pos_after = get_world_point(
                env.render_env.obj_id,
                env.render_env.link_name_to_index[target_link],
                gripper_object_contact_local,
            )

            # Now with filter: we guarantee that every step is correct!!
            delta_gripper = np.array(gripper_tip_pos_after) - np.array(
                gripper_tip_pos_before
            )
            last_step_grasp_point = best_point

            update_history_signal = False
            # -----------Update the direction and history stack!!!!-----------
            if len(correct_direction_stack) == 0:
                # Update direction stack
                if np.linalg.norm(delta_gripper) > initial_movement_thres:
                    correct_direction_stack.append(
                        delta_gripper / (np.linalg.norm(delta_gripper) + 1e-12)
                    )
                # Update history stack
                if (
                    not history_filter
                    or np.linalg.norm(delta_gripper) > good_movement_thres
                ):
                    update_history_signal = True
                    use_history = True
                    prev_flow_pred = pred_flow.clone()  # History flow
                    prev_point_cloud = copy.deepcopy(P_world)  # History point cloud
            else:
                # Update direction stack:
                if np.dot(delta_gripper, correct_direction_stack[-1]) > 0:  # Consistent
                    correct_direction_stack.append(
                        delta_gripper / (np.linalg.norm(delta_gripper) + 1e-12)
                    )
                    if (
                        not history_filter
                        or np.linalg.norm(delta_gripper) > good_movement_thres
                    ):
                        update_history_signal = True
                        prev_flow_pred = pred_flow.clone()  # History flow
                        prev_point_cloud = copy.deepcopy(P_world)  # History point cloud
            update_history_step.append(update_history_signal)
        else:  # Reset
            if history_filter:
                use_history = False
                update_history_step.append(0)
            else:  # no history filter: always update history
                use_history = True
                update_history_step.append(1)
                prev_flow_pred = pred_flow.clone()  # History flow
                prev_point_cloud = copy.deepcopy(P_world)  # History point cloud
            last_step_grasp_point = None
        global_step += 1
        this_step_trial = 0

        if website:
            # Add pcd to flow animation
            segmented_flow = np.zeros_like(pred_flow)
            segmented_flow[link_ixs] = pred_flow[link_ixs]
            segmented_flow = np.array(
                normalize_trajectory(
                    torch.from_numpy(np.expand_dims(segmented_flow, 1))
                ).squeeze()
            )
            animation.add_trace(
                torch.as_tensor(P_world),
                torch.as_tensor([P_world]),
                torch.as_tensor([segmented_flow * 3]),
                "red",
            )

            # Capture frame
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
            # rgbImgOpenCV = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
            # videoWriter.write(rgbImgOpenCV)
            image = np.array(rgbImg, dtype=np.uint8)
            image = image[:, :, :3]

            # Add the frame to the video
            writer.append_data(image)

        # breakpoint()
        success, sim_trajectory[global_step] = env.detect_success(target_link)

        if success:
            break

        # pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)   # Render a new point cloud!
        # if len(correct_direction_stack) == 2:
        #     breakpoint()

    for left_step in range(global_step, 31):
        sim_trajectory[left_step] = sim_trajectory[global_step]
    # calculate the metrics
    # if success:
    #     metric = 1
    # else:
    #     curr_pos = env.get_joint_value(target_link)
    #     metric = (curr_pos - init_angle) / (target_angle - init_angle)
    #     metric = min(max(metric, 0), 1)
    curr_pos = env.get_joint_value(target_link)
    metric = (curr_pos - init_angle) / (target_angle - init_angle)
    metric = min(max(metric, 0), 1)

    if website:
        if gui:
            p.stopStateLogging(log_id)
        else:
            writer.close()
            # videoWriter.release()

    p.disconnect(physicsClientId=env.render_env.client_id)
    animation_results = None if not website else animation.animate()
    return (
        animation_results,
        TrialResult(  # Save the flow visuals
            success=success,
            contact=True,
            assertion=True,
            init_angle=init_angle,
            final_angle=target_angle,
            now_angle=curr_pos,
            metric=metric,
        ),
        sim_trajectory
        if not analysis
        else [
            sim_trajectory,
            update_history_step,
            cc_cnts,
            sgp_signals,
            visual_all_points,
            visual_link_ixs,
            visual_grasp_points_idx,
            visual_grasp_points,
            visual_flows,
        ],
    )


def run_trial_with_switch_models(
    env: PMSuctionSim,
    raw_data: PMObject,
    target_link: str,
    model,
    switch_model,
    history_for_models,
    gt_model=None,  # When we use mask_input_channel=True, this is the mask generator
    n_steps: int = 30,
    n_pts: int = 1200,
    save_name: str = "unknown",
    website: bool = False,
    gui: bool = False,
    return_switch_ids: bool = False,
) -> TrialResult:
    torch.manual_seed(42)
    torch.set_printoptions(precision=10)  # Set higher precision for PyTorch outputs
    np.set_printoptions(precision=10)
    # p.setPhysicsEngineParameter(numSolverIterations=10)
    # p.setPhysicsEngineParameter(contactBreakingThreshold=0.01, contactSlop=0.001)

    sim_trajectory = [0.0] + [0] * (n_steps)  # start from 0.05
    model_ids = [-1] + [0] * (n_steps)

    if website:
        # Flow animation
        animation = FlowNetAnimation()

    # First, reset the environment.
    env.reset()
    # Joint information
    info = p.getJointInfo(
        env.render_env.obj_id,
        env.render_env.link_name_to_index[target_link],
        env.render_env.client_id,
    )
    init_angle, target_angle = info[8], info[9]

    # Sometimes doors collide with themselves. It's dumb.
    if (
        raw_data.category == "Door"
        and raw_data.semantics.by_name(target_link).type == "hinge"
    ):
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.2)

    if raw_data.semantics.by_name(target_link).type == "hinge":
        env.set_joint_state(target_link, init_angle + 0.0 * (target_angle - init_angle))
        # env.set_joint_state(target_link, 0.05)

    # Predict the flow on the observation.
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

    if init_angle == target_angle:  # Not movable
        p.disconnect(physicsClientId=env.render_env.client_id)
        return (
            None,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory if not return_switch_ids else (sim_trajectory, model_ids),
        )

    # breakpoint()
    if gt_model is None:  # No mask
        # For the first step, never use history!!!
        pred_trajectory = model(copy.deepcopy(pc_obs))
    else:
        movable_mask = gt_model.get_movable_mask(pc_obs)
        pred_trajectory = model(copy.deepcopy(pc_obs), movable_mask)
    pred_trajectory = pred_trajectory.reshape(
        pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
    )
    traj_len = pred_trajectory.shape[1]  # Trajectory length
    print(f"Predicting {traj_len} length trajectories.")
    pred_flow = pred_trajectory[:, 0, :]

    # Filter down just the points on the target link.
    link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
    # assert link_ixs.any()
    if not link_ixs.any():
        p.disconnect(physicsClientId=env.render_env.client_id)
        print("link_ixs finds no point")
        animation_results = animation.animate() if website else None
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=False,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory if not return_switch_ids else (sim_trajectory, model_ids),
        )

    if website:
        if gui:
            # Record simulation video
            log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                f"./logs/simu_eval/video_assets/{save_name}.mp4",
            )
        else:
            video_file = f"./logs/simu_eval/video_assets/{save_name}.mp4"
            # # cv2 output videos won't show on website
            frame_width = 640
            frame_height = 480
            # fps = 5
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
            # videoWriter.write(rgbImgOpenCV)

            # Camera param
            writer = imageio.get_writer(video_file, fps=5)

            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                width=frame_width,
                height=frame_height,
                viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=[0, 0, 0],
                    distance=5,
                    # yaw=180,
                    yaw=270,
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
            writer.append_data(image)

    # The attachment point is the point with the highest flow.
    # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
    best_flow_ix, best_flows, best_points = choose_grasp_points(
        pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=20
    )

    # Teleport to an approach pose, approach, the object and grasp.
    if website and not gui:
        # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
        best_flow_ix, contact = env.teleport(
            best_points, best_flows, video_writer=writer
        )
    else:
        # contact = env.teleport_and_approach(best_point, best_flow)
        best_flow_ix, contact = env.teleport(best_points, best_flows)

    prev_flow_pred = pred_flow.clone()  # History flow
    prev_point_cloud = copy.deepcopy(P_world)  # History point cloud

    best_flow = pred_flow[link_ixs][best_flow_ix]
    best_point = P_world[link_ixs][best_flow_ix]
    last_step_grasp_point = best_point

    if not contact:
        if website:
            segmented_flow = np.zeros_like(pred_flow)
            segmented_flow[link_ixs] = pred_flow[link_ixs]
            segmented_flow = np.array(
                normalize_trajectory(
                    torch.from_numpy(np.expand_dims(segmented_flow, 1))
                ).squeeze()
            )
            animation.add_trace(
                torch.as_tensor(P_world),
                torch.as_tensor([P_world]),
                torch.as_tensor([segmented_flow * 3]),
                "red",
            )
            if gui:
                p.stopStateLogging(log_id)
            else:
                # Write video
                writer.close()
                # videoWriter.release()

        print("No contact!")
        p.disconnect(physicsClientId=env.render_env.client_id)
        animation_results = None if not website else animation.animate()
        return (
            animation_results,
            TrialResult(
                success=False,
                assertion=True,
                contact=False,
                init_angle=0,
                final_angle=0,
                now_angle=0,
                metric=0,
            ),
            sim_trajectory if not return_switch_ids else (sim_trajectory, model_ids),
        )

    env.attach()
    # breakpoint()
    pc_obs = env.render(filter_nonobj_pts=True, n_pts=n_pts)
    success = False

    should_switch_model = False

    global_step = 0
    # for i in range(n_steps):
    while global_step < n_steps:
        # Predict the flow on the observation.
        if gt_model is None:  # GT Flow model
            if should_switch_model:
                print("Using model 2!")
                # Use history model
                if history_for_models[1]:
                    pred_trajectory = switch_model(
                        copy.deepcopy(pc_obs),
                        copy.deepcopy(prev_point_cloud),
                        copy.deepcopy(prev_flow_pred.numpy()),
                    )
                else:  # Switch model is not a history_based_model
                    pred_trajectory = switch_model(copy.deepcopy(pc_obs))
            else:
                pred_trajectory = model(copy.deepcopy(pc_obs))
        else:
            movable_mask = gt_model.get_movable_mask(pc_obs)
            # breakpoint()
            pred_trajectory = model(pc_obs, movable_mask)
            # pred_trajectory = model(pc_obs)
        pred_trajectory = pred_trajectory.reshape(
            pred_trajectory.shape[0], -1, pred_trajectory.shape[-1]
        )

        for traj_step in range(pred_trajectory.shape[1]):
            if global_step == n_steps:
                break
            global_step += 1
            pred_flow = pred_trajectory[:, traj_step, :]
            rgb, depth, seg, P_cam, P_world, pc_seg, segmap = pc_obs

            # Filter down just the points on the target link.
            # breakpoint()
            link_ixs = pc_seg == env.render_env.link_name_to_index[target_link]
            # assert link_ixs.any()
            if not link_ixs.any():
                if website:
                    if gui:
                        p.stopStateLogging(log_id)
                    else:
                        writer.close()
                        # videoWriter.release()
                p.disconnect(physicsClientId=env.render_env.client_id)
                print("link_ixs finds no point")
                animation_results = animation.animate() if website else None
                return (
                    animation_results,
                    TrialResult(
                        assertion=False,
                        success=False,
                        contact=False,
                        init_angle=0,
                        final_angle=0,
                        now_angle=0,
                        metric=0,
                    ),
                    sim_trajectory
                    if not return_switch_ids
                    else (sim_trajectory, model_ids),
                )

            # Get the best direction.
            # best_flow_ix = pred_flow[link_ixs].norm(dim=-1).argmax()
            best_flow_ix, best_flows, best_points = choose_grasp_points(
                pred_flow[link_ixs], P_world[link_ixs], filter_edge=False, k=20
            )

            # (1) Strategy 1 - Don't change grasp point
            # (2) Strategy 2 - Change grasp point when leverage difference is large
            lev_diff_thres = 0.2
            no_movement_thres = -1

            # # Don't use this policy
            # lev_diff_thres = 100
            # no_movement_thres = -1
            # good_movement_thres = 1000

            # Only change if the new point's leverage is a great increase
            # gripper_tip_pos = p.getClosestPoints(
            #     env.gripper.body_id, env.render_env.obj_id, distance=0.5, linkIndexA=0
            # )[0][5]
            # gripper_object_contact = p.getContactPoints(
            #     env.gripper.body_id, env.render_env.obj_id, linkIndexA=0
            # )[0]
            # gripper_contact, object_contact = gripper_object_contact[5], gripper_object_contact[6]
            gripper_tip_pos, _ = p.getBasePositionAndOrientation(env.gripper.body_id)
            pcd_dist = torch.tensor(P_world[link_ixs] - np.array(gripper_tip_pos)).norm(
                dim=-1
            )
            grasp_point_id = pcd_dist.argmin()
            lev_diff = best_flows.norm(dim=-1) - pred_flow[link_ixs][
                grasp_point_id
            ].norm(dim=-1)

            gripper_movement = torch.from_numpy(
                P_world[link_ixs][grasp_point_id] - last_step_grasp_point
            ).norm()
            # print("gripper: ",gripper_movement)
            # breakpoint()
            if (
                gripper_movement < no_movement_thres or lev_diff[0] > lev_diff_thres
            ):  # pcd_dist < 0.05 -> didn't move much....
                env.reset_gripper(target_link)
                p.stepSimulation(
                    env.render_env.client_id
                )  # Make sure the constraint is lifted

                if website and not gui:
                    # contact = env.teleport_and_approach(best_point, best_flow, video_writer=writer)
                    best_flow_ix, contact = env.teleport(
                        best_points, best_flows, video_writer=writer
                    )
                else:
                    # contact = env.teleport_and_approach(best_point, best_flow)
                    best_flow_ix, contact = env.teleport(best_points, best_flows)
                best_flow = pred_flow[link_ixs][best_flow_ix]
                best_point = P_world[link_ixs][best_flow_ix]
                last_step_grasp_point = best_point  # Grasp a new point
                # print("new!", last_step_grasp_point)

                if not contact:
                    if website:
                        segmented_flow = np.zeros_like(pred_flow)
                        segmented_flow[link_ixs] = pred_flow[link_ixs]
                        segmented_flow = np.array(
                            normalize_trajectory(
                                torch.from_numpy(np.expand_dims(segmented_flow, 1))
                            ).squeeze()
                        )
                        animation.add_trace(
                            torch.as_tensor(P_world),
                            torch.as_tensor([P_world]),
                            torch.as_tensor([segmented_flow * 3]),
                            "red",
                        )
                        if gui:
                            p.stopStateLogging(log_id)
                        else:
                            # Write video
                            writer.close()
                            # videoWriter.release()

                    print("No contact!")
                    p.disconnect(physicsClientId=env.render_env.client_id)
                    animation_results = None if not website else animation.animate()
                    return (
                        animation_results,
                        TrialResult(
                            success=False,
                            assertion=True,
                            contact=False,
                            init_angle=0,
                            final_angle=0,
                            now_angle=0,
                            metric=0,
                        ),
                        sim_trajectory
                        if not return_switch_ids
                        else (sim_trajectory, model_ids),
                    )

                env.attach()
            else:
                best_flow = pred_flow[link_ixs][best_flow_ix[0]]
                last_step_grasp_point = P_world[link_ixs][
                    grasp_point_id
                ]  # The original point - don't need to change
                # print("same:", last_step_grasp_point)

            env.attach()
            # Perform the pulling.
            # if best_flow.sum() == 0:
            #     continue
            # print(best_flow)
            env.pull(best_flow)
            env.attach()

            if website:
                # Add pcd to flow animation
                segmented_flow = np.zeros_like(pred_flow)
                segmented_flow[link_ixs] = pred_flow[link_ixs]
                segmented_flow = np.array(
                    normalize_trajectory(
                        torch.from_numpy(np.expand_dims(segmented_flow, 1))
                    ).squeeze()
                )
                animation.add_trace(
                    torch.as_tensor(P_world),
                    torch.as_tensor([P_world]),
                    torch.as_tensor([segmented_flow * 3]),
                    "red",
                )

                # Capture frame
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                    width=frame_width,
                    height=frame_height,
                    viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 0, 0],
                        distance=5,
                        yaw=270,
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
                # rgbImgOpenCV = cv2.cvtColor(np.array(rgbImg), cv2.COLOR_RGB2BGR)
                # videoWriter.write(rgbImgOpenCV)
                image = np.array(rgbImg, dtype=np.uint8)
                image = image[:, :, :3]

                # Add the frame to the video
                writer.append_data(image)

            success, sim_trajectory[global_step] = env.detect_success(target_link)
            model_ids[global_step] = should_switch_model

            if success:
                for left_step in range(global_step, 31):
                    sim_trajectory[left_step] = sim_trajectory[global_step]
                    model_ids[left_step] = model_ids[global_step]
                break

            # Previous step
            # # Policy - 1
            # should_switch_model = True  # Always use history when there is history

            # # Policy - 2
            # should_switch_model = (  # If last step makes progress
            #     sim_trajectory[global_step] - sim_trajectory[global_step - 1]
            # ) > 0.01

            # Policy - 3
            should_switch_model = (sim_trajectory[global_step]) > 0.05  # If it's opened

            # # Policy - 4
            # should_switch_model = (  # If last step makes progress
            #     sim_trajectory[global_step]
            # ) > 0.1 and (sim_trajectory[global_step] - sim_trajectory[global_step - 1]) > 0.01

            prev_flow_pred = pred_flow.clone()
            prev_point_cloud = copy.deepcopy(pc_obs[4])
            pc_obs = env.render(filter_nonobj_pts=True, n_pts=1200)

        if success:
            for left_step in range(global_step, 31):
                sim_trajectory[left_step] = sim_trajectory[global_step]
            break

    # calculate the metrics
    curr_pos = env.get_joint_value(target_link)
    metric = (curr_pos - init_angle) / (target_angle - init_angle)
    metric = min(max(metric, 0), 1)

    if website:
        if gui:
            p.stopStateLogging(log_id)
        else:
            writer.close()
            # videoWriter.release()

    p.disconnect(physicsClientId=env.render_env.client_id)
    animation_results = None if not website else animation.animate()
    return (
        animation_results,
        TrialResult(  # Save the flow visuals
            success=success,
            contact=True,
            assertion=True,
            init_angle=init_angle,
            final_angle=target_angle,
            now_angle=curr_pos,
            metric=metric,
        ),
        sim_trajectory if not return_switch_ids else (sim_trajectory, model_ids),
    )
