import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import cv2
import gym
import os
import open3d as o3d
import numpy.typing as npt
import xml.etree.ElementTree as ET
from utils.tracker import Tracker

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import rpad
from rpad.pybullet_libs.camera import Camera

class ScoopEnv(gym.Env):
    def __init__(self,
                 panda_start_pos = [0,0,0],
                 panda_start_orientation = [0,0,0],
                 panda_initial_positions = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0, 0, 0, 0],
                 spatula_pos = [0.6, 0, 0.02],
                 spatula_orientation = [math.radians(90), 0, math.radians(90)],
                 rod_orientation = [0, 0, math.radians(90)],
                 rod_id = 'knife',
                 ikMaxNumIterations = 100,
                 use_GUI = False,
                 data_path = "~/datasets/unevenobject/raw/train"
                 ):
        super(ScoopEnv, self).__init__()
        
        # Connect to PyBullet
        if use_GUI:
            self.physicsClientId = p.connect(p.GUI)
        else:
            self.physicsClientId = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        self.asset_path = pybullet_data.getDataPath() if data_path is None else os.path.expanduser(data_path)
        self.pybullet_data_path = pybullet_data.getDataPath()
        # print(p.getPhysicsEngineParameters())

        self.timesteps = 0
        self.fps = 30 # pybullet envs actually work at 240Hz
        self.ikMaxNumIterations = ikMaxNumIterations
        self.last_action = None

        self.planeId = p.loadURDF(os.path.join(self.pybullet_data_path, "plane.urdf"), physicsClientId=self.physicsClientId)

        # Load the Franka Panda robot
        self.pandaStartPos = panda_start_pos
        self.pandaStartOrientation = p.getQuaternionFromEuler(panda_start_orientation)
        self.pandaUid = p.loadURDF(os.path.join(self.pybullet_data_path, "franka_panda/panda_with_spatula.urdf"), self.pandaStartPos, self.pandaStartOrientation, useFixedBase=True)
        self.spatulaUid = self.get_link_id("spatula") # spatula is ((0.40096041891694134, -0.03455015056360143, 0.001526565061983419), (0.6959624711138597, 0.034601380122758776, 0.09075669095962141))
        
        self.numJoints = p.getNumJoints(self.pandaUid)
        self.availableJointsIndexes = [i for i in range(p.getNumJoints(self.pandaUid)) if p.getJointInfo(self.pandaUid, i)[2] != p.JOINT_FIXED]
        self.pandaInitialPosition = panda_initial_positions

        # self.print_joints()

        # Load Rod and its supporting cuboids
        self.rod_id = rod_id
        self.rodPath = os.path.join(self.asset_path, rod_id, "rod.urdf")
        self.rodShape, self.rodMassPosition = self._get_urdf_shape(self.rodPath)
        self.cuboidPath = os.path.join(self.pybullet_data_path, "cuboid.urdf")
        self.cuboidShape, _ = self._get_urdf_shape(self.cuboidPath)
        
        self.rodPosition = spatula_pos.copy()
        # print(self.rodShape, self.rodMassPosition)
        self.rodPosition = [spatula_pos[0] + 0.05 + self.rodMassPosition[1],
                            0 + self.rodMassPosition[0], 
                            self.cuboidShape[2] + self.rodShape[2]/2]
        self.rodOrientation = p.getQuaternionFromEuler(rod_orientation)
        self.rodEuler = rod_orientation
        # print(self.rodPosition, self.rodOrientation)
        self.rodUid = p.loadURDF(self.rodPath, basePosition=self.rodPosition, baseOrientation=self.rodOrientation, physicsClientId=self.physicsClientId)
        self.cuboidPositionY = (self.rodShape[0]/2)*0.9
        self.supporting_cuboid_1 = p.loadURDF(self.cuboidPath, basePosition = [self.rodPosition[0], self.cuboidPositionY, 0], useFixedBase=True, physicsClientId=self.physicsClientId)
        self.supporting_cuboid_2 = p.loadURDF(self.cuboidPath, basePosition = [self.rodPosition[0], -self.cuboidPositionY, 0], useFixedBase=True, physicsClientId=self.physicsClientId)
        
        self.right_most = self.cuboidPositionY - self.cuboidShape[1]/2 - 0.0346
        self.left_most = -self.right_most
        # print(self.rodShape, self.cuboidShape, self.left_most, self.right_most)
        # print(p.getBasePositionAndOrientation(int(self.rodUid)))
        
        self.scale_factors=None
        self.pcd_center=None
            
        self.link_name_to_index = {
            p.getBodyInfo(self.rodUid, physicsClientId=self.physicsClientId)[0].decode(
                "UTF-8"
            ): -1,
        }
        self.jn_to_ix = {}

        # Get the segmentation.
        for _id in range(p.getNumJoints(self.rodUid, physicsClientId=self.physicsClientId)):
            info = p.getJointInfo(self.rodUid, _id, physicsClientId=self.physicsClientId)
            joint_name = info[1].decode("UTF-8")
            link_name = info[12].decode("UTF-8")
            self.link_name_to_index[link_name] = _id

            # Only store if the joint is one we can control.
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                self.jn_to_ix[joint_name] = _id

        # self.disable_collision()
        self.init_camera()
        
        self.tracker = Tracker(self.camera)

        p.resetDebugVisualizerCamera(cameraDistance=2.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0])
    

    def init_reset(self):
        p.setGravity(0,0,-10)
        p.setPhysicsEngineParameter(numSolverIterations=50)  # Increase solver iterations
        p.setTimeStep(1/240.0)  # Use a smaller time step

        # reset franka
        p.changeDynamics(self.pandaUid, 9, lateralFriction=5, spinningFriction=5, rollingFriction=5)
        p.changeDynamics(self.pandaUid, 10, lateralFriction=5, spinningFriction=5, rollingFriction=5)
        for joint_index in self.availableJointsIndexes:
            p.resetJointState(self.pandaUid, joint_index, self.pandaInitialPosition[joint_index])
        
        self.change_collision_with_object(turn_on=False)
        # Obtain the default spatula orientation
        link_state = p.getLinkState(self.pandaUid, self.spatulaUid)
        self.spatula_orientation = link_state[1]  # (x, y, z, w)
        
        # self.print_joints()
        pos, orn = p.getBasePositionAndOrientation(self.rodUid)
        self.rodOrientation = orn
        # print(self.rodOrientation)
        xyz_offset = p.getAABB(self.rodUid, physicsClientId=self.physicsClientId)
        self.x_offset = (xyz_offset[1][0] - xyz_offset[0][0])/2
        self.y_offset = (xyz_offset[1][1] - xyz_offset[0][1])/2
        self.z_offset = (xyz_offset[1][2] - xyz_offset[0][2])/2
        self.cuboidPositionY = (self.y_offset)*0.9
        self.right_most = self.cuboidPositionY - self.cuboidShape[1]/2 - 0.0346 # 0.021
        self.left_most = -self.right_most
        p.resetBasePositionAndOrientation(
            self.rodUid,
            posObj=[self.rodPosition[0],
                    self.rodPosition[1],
                    self.cuboidShape[2] + self.z_offset],
            ornObj=self.rodOrientation,
            physicsClientId=self.physicsClientId,
        )
        p.resetBasePositionAndOrientation(
            self.supporting_cuboid_1,
            posObj=[self.rodPosition[0],
                    self.cuboidPositionY, 
                    self.cuboidShape[2]/2],
            ornObj=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physicsClientId,
        )
        p.resetBasePositionAndOrientation(
            self.supporting_cuboid_2,
            posObj=[self.rodPosition[0],
                    -self.cuboidPositionY, 
                    self.cuboidShape[2]/2],
            ornObj=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physicsClientId,
        )
        self.T_world_base = np.eye(4)
        self.T_world_base[2, 3] = self.cuboidShape[2]-self.z_offset
        
        # p.setJointMotorControl2(
        #     self.rodUid,
        #     0,  # 关节索引（从 0 开始）
        #     p.POSITION_CONTROL,
        #     targetPosition=0.5,
        #     force=100  # 最大驱动力
        # )
        # # 创建约束：将物体与世界坐标系绑定，仅允许绕 X 轴旋转
        # constraint_id = p.createConstraint(
        #     parentBodyUniqueId=self.rodUid,
        #     parentLinkIndex=-1,          # -1 表示基座（base link）
        #     childBodyUniqueId=-1,        # -1 表示世界坐标系
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,     # 初始固定所有自由度
        #     jointAxis=[1, 0, 0],         # 允许绕 X 轴旋转（Pitch）
        #     parentFramePosition=[0, 0, 0],
        #     childFramePosition=[0, 0, 0],
        #     # jointLowerLimit=0,           # 最小旋转角度（弧度）
        #     # jointUpperLimit=0,           # 最大旋转角度（初始设为 0，需后续动态调整）
        #     # erp=0.8,                     # 位置控制增益
        #     # maxForce=500                 # 最大约束力
        # )

        # # 允许绕 X 轴旋转，但禁止其他自由度
        # p.changeConstraint(
        #     constraint_id,
        #     jointLowerLimit=-3.14,       # 允许绕 X 轴旋转的最小角度（-π）
        #     jointUpperLimit=3.14,         # 允许绕 X 轴旋转的最大角度（π）
        #     maxForce=500
        # )
    
    def reset(self):
        self.change_collision_with_object(turn_on=False)
        p.resetBasePositionAndOrientation(
            self.rodUid,
            posObj=[self.rodPosition[0],
                    self.rodPosition[1],
                    self.cuboidShape[2] + self.z_offset],
            ornObj=self.rodOrientation,
            physicsClientId=self.physicsClientId,
        )
        
    
    def step(self, action, record = False, writer = None):
        success = self.step_action(action, record, writer)
        return success
        # print(p.getBasePositionAndOrientation(self.spatulaUid))
        # print(p.getBasePositionAndOrientation(self.rodUid))
    
    def check_reset(self):
        contacts = p.getContactPoints(self.planeId, self.rodUid)
        if(len(contacts)>0):
            print('reset')
            return True
        base_position, base_orientation = p.getBasePositionAndOrientation(self.rodUid)
        if base_position[2] < self.cuboidShape[2]: #  - 0.01:
            print('below cuboid reset')
            return True
        return False
    
    def render(self, filter_nonobj_pts=False, n_pts=None, normalize_pcd=False, remove_cuboid=True): # TODO: remove cuboid by clipping z-axis
        render = {}
        obs = self.camera.render(self.physicsClientId)
        obs.pop('P_rgb')
        obs["rgb"] = cv2.cvtColor(obs["rgb"][:, :, :3], cv2.COLOR_RGB2BGR)
        if remove_cuboid:
            idx=(obs["P_world"][:,2]>=0.065)
            obs["P_world"] = obs["P_world"][idx]
            obs["pc_seg"] = obs["pc_seg"][idx]
            
        render["P_world_org"] = obs["P_world"]
        
        if filter_nonobj_pts:
            P_world = obs["P_world"]
            pc_seg = obs["pc_seg"]
            segmap = obs["segmap"]
            
            # Reindex the segmentation.
            pc_seg_obj = np.ones_like(pc_seg) * 0
            for k, (body, link) in segmap.items():
                if body in [self.rodUid, self.supporting_cuboid_1, self.supporting_cuboid_2]:
                    ixs = pc_seg == k
                    pc_seg_obj[ixs] = 1

            obs["P_world"] = P_world[pc_seg_obj==1]
            obs["pc_seg"] = pc_seg[pc_seg_obj==1]
            render["P_world_org"] = obs["P_world"]
        
        if normalize_pcd:
            normalized_pc, scale_factors, pcd_center = self.normalize_pcd(obs["P_world"])
            obs["P_world"] = normalized_pc
            self.scale_factors=scale_factors
            self.pcd_center=pcd_center
        
        if n_pts:
            P_world_org = render["P_world_org"]
            P_world = obs["P_world"]
            pc_seg = obs['pc_seg']
            if len(P_world) < n_pts:
                repeat_times = n_pts // len(P_world) + 1
                P_world = np.tile(P_world, (repeat_times, 1))
                P_world_org = np.tile(P_world_org, (repeat_times, 1))
                pc_seg = np.tile(pc_seg, (repeat_times))
                
            rng = np.random.default_rng(42) # magic number
            seed1, seed2 = rng.bit_generator._seed_seq.spawn(2)
            rng = np.random.default_rng(seed2)
            ixs = rng.permutation(range(len(P_world)))[: n_pts]
            render["P_world_org"] = P_world_org[ixs]
            obs["P_world"] = P_world[ixs]
            obs["pc_seg"] = pc_seg[ixs]
        
        pc_seg = obs["pc_seg"]
        segmap = obs["segmap"]
        # Reindex the segmentation.
        pc_seg_obj = np.ones_like(pc_seg) * 0
        for k, (body, link) in segmap.items():
            if body == self.rodUid:
                ixs = pc_seg == k
                pc_seg_obj[ixs] = 1

        # P_world_rod = P_world[pc_seg_obj==1]
        is_lifted, is_steady = self.check_lifted()
        render["lifted"] = is_lifted
        render["steady"] = is_steady
        render["P_world"] = obs["P_world"]
        render["rgb"] = obs["rgb"]
        render["obs"] = obs
        render["pc_seg_obj"] = pc_seg_obj
        
        return render
    
    def close(self):
        p.disconnect(self.physicsClientId)
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def get_link_id(self, joint_name):  # Find the spatula link
        num_joints = p.getNumJoints(self.pandaUid)
        
        for joint_id in range(num_joints):
            joint_info = p.getJointInfo(self.pandaUid, joint_id)
            # print(joint_info[12].decode('utf-8'))
            if joint_info[12].decode('utf-8') == joint_name:
                return joint_info[0]  # Joint ID corresponds to link index
        return None
    
    def change_collision_with_object(self, turn_on=True):
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.pandaUid, 
            bodyUniqueIdB=self.rodUid,
            linkIndexA=self.spatulaUid,
            linkIndexB=-1,
            enableCollision=1 if turn_on else 0,
            physicsClientId=self.physicsClientId
        )

    def disable_collision(self):
        # Disable all of franka joints with the rod
        num_links = p.getNumJoints(self.pandaUid)
        for gripper_link_id in range(-1, num_links):
            # if gripper_link_id != self.spatulaUid:
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.pandaUid, 
                bodyUniqueIdB=self.rodUid,
                linkIndexA=gripper_link_id,
                linkIndexB=-1,
                enableCollision=0,
                physicsClientId=self.physicsClientId
            )

        # Disable collision between spatula & supporting cuboid
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.pandaUid, 
            bodyUniqueIdB=self.supporting_cuboid_1,
            linkIndexA=self.spatulaUid,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.physicsClientId
        )

        p.setCollisionFilterPair(
            bodyUniqueIdA=self.pandaUid, 
            bodyUniqueIdB=self.supporting_cuboid_2,
            linkIndexA=self.spatulaUid,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.physicsClientId
        )

        # Disable collision between spatula & floor
        p.setCollisionFilterPair(
            bodyUniqueIdA=self.pandaUid,
            bodyUniqueIdB=self.planeId,
            linkIndexA=self.spatulaUid,
            linkIndexB=-1,
            enableCollision=0,
            physicsClientId=self.physicsClientId
        )
            
    
    def print_joints(self):
        for i in range(self.numJoints):
            info = p.getJointInfo(self.pandaUid, i)
            link_name = info[12].decode('utf-8')
            joint_limits = info[8:10]
            joint_name = info[1].decode('utf-8')
            print("joint id: ", i)
            print("link_name: ", link_name)
            print("joint_name: ", joint_name)
            print("joint_limits: ", joint_limits)
            print("joint type: ", info[2])
            print("joint limit: ", info[8:10])
            print("===================")

        # Spatula information
        link_state = p.getLinkState(self.pandaUid, self.spatulaUid)
        link_position = link_state[0]  # (x, y, z)
        link_orientation = link_state[1]  # (x, y, z, w)
        print("link position: ", link_position)
        print("link orientation: ", link_orientation)
        print("===================")

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

    def init_video_writer(self, video_path):
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, self.fourcc, self.fps, (self.frame_width, self.frame_height))
    
    def step_action(self, action, record, writer = None):
        # print(action)
        target_position = action[:3]
        # target_orientation = action[3:6] 
        # target_gripper = action[6:]

        lower_limits = []
        upper_limits = []

        for joint_id in self.availableJointsIndexes:
            joint_info = p.getJointInfo(self.pandaUid, joint_id)
            if joint_id == 9 or joint_id == 10:
                lower_limits.append(0)
                upper_limits.append(0)
            else:
                lower_limits.append(joint_info[8])  # Lower limit
                upper_limits.append(joint_info[9])  # Upper limit

        jointPoses = p.calculateInverseKinematics(
                self.pandaUid, endEffectorLinkIndex = self.spatulaUid, 
                targetPosition=target_position, targetOrientation=self.spatula_orientation, 
                lowerLimits=lower_limits, upperLimits=upper_limits, 
                maxNumIterations=self.ikMaxNumIterations)  


        for _ in range(self.ikMaxNumIterations*2):
            p.setJointMotorControlArray(
                bodyUniqueId=self.pandaUid,
                jointIndices=self.availableJointsIndexes,
                controlMode=p.POSITION_CONTROL,
                targetPositions=jointPoses,
            )
            # p.setJointMotorControlArray(self.pandaUid, [9, 10], p.POSITION_CONTROL, targetPositions=target_gripper)

            current_position, current_orientation = p.getLinkState(self.pandaUid, self.spatulaUid)[0:2]
            position_error = np.linalg.norm(np.array(target_position) - np.array(current_position))
            # orientation_error = np.linalg.norm(np.array(target_orientation) - np.array(current_orientation))
            # print(current_position, target_position, position_error)
            actual_joint_states = [
                p.getJointState(self.pandaUid, i)[0]  # Extract joint position
                for i in range(p.getNumJoints(self.pandaUid))
                if p.getJointInfo(self.pandaUid, i)[2] != p.JOINT_FIXED
            ]
            
            pos, orn = p.getBasePositionAndOrientation(self.rodUid)
            euler = p.getEulerFromQuaternion(orn)
            if abs(euler[0] - self.rodEuler[0]) > 0.0001:
                new_euler = (self.rodEuler[0], euler[1], self.rodEuler[2])
                new_orn = p.getQuaternionFromEuler(new_euler)
                p.resetBasePositionAndOrientation(self.rodUid, pos, new_orn)
            
            # print(self.availableJointsIndexes)
            # print("Actual joint:", actual_joint_states, "Target joint:", jointPoses)
            if position_error < 0.001:  # Adjust thresholds if needed
                # print("Approached to the desired point!")
                return True  # Stop if close enough

            p.stepSimulation()
            if record:
                if writer is None:
                    self.video_writer.write(self.render()["rgb"])
                else:
                    writer.append_data(self.render()["rgb"])
            # time.sleep(1/self.fps)
        # print("fail to Aprroach to desired point")
        return False
    
    def prepare_scoop(self):
        # Open the gripper and move the gripper above the spatula
        action = [0.55, 0, 0.045] + [None]*3 + [0.0, 0.0]
        _ = self.step(action)

        return action

    def check_steady(self, threshold=0.01):
        _, base_orientation = p.getBasePositionAndOrientation(self.rodUid)
        euler_angles = p.getEulerFromQuaternion(base_orientation)

        roll = euler_angles[0]
        pitch = euler_angles[1]
        
        # if abs(roll) > threshold:
        #     print(f"Warning! roll is {roll:.4g}")

        is_parallel_to_ground = abs(pitch) < threshold

        # print(f"Roll: {roll:.4g}, Pitch: {pitch:.4g}")
        # print(f"Is parallel to ground: {is_parallel_to_ground}")
        return is_parallel_to_ground

    def check_lifted(self, steady_threshold=0.01, lift_threashold=0.015):
        # return False, False # Debug without rod
        base_position, _ = p.getBasePositionAndOrientation(self.rodUid)
        is_steady = self.check_steady(steady_threshold)
        is_lifted = is_steady and base_position[2] > lift_threashold + self.rodPosition[2]
        return is_lifted, is_steady
    
    def detect_success(self): # , best_point=None
        is_lifted, is_steady = self.check_lifted()
        success = is_lifted and is_steady
        # if best_point is None:
        #     metric = None
        # else:
            # gt_y = self.rodMassPosition[0] + 1e-6
            # selected_y = best_point[1]
            # length = self.rodShape[0]
            # metric = selected_y / gt_y
            # metric = min(max(metric, -1), 1)
        metric = self.get_score()
        return success, metric
        
    def choose_action_random(self, last_action, start_step=False):
        action = last_action.copy()
        
        action[1] = np.random.uniform(self.left_most, self.right_most)
        
        if start_step:
            # print('start_step')
            gt_y = self.rodMassPosition[0]
            while action[1] <= gt_y + 0.1 and action[1] >= gt_y - 0.1:
                action[1] = np.random.uniform(self.left_most, self.right_most)
                # print(action[1], gt_y)
        # print(f'action: {action}')
        return action

    def choose_action_random_uniform(self, last_action, i, total=10): # divide into i parts [0,1]...[i-1,i]
        action = last_action.copy()
        rod_length = self.y_offset*2
        
        interval = (0.9*rod_length-0.05)/total
        left_most = self.left_most + interval*i
        right_most = self.left_most + interval*(i+1)
        
        action[1] = np.random.uniform(left_most, right_most)
        
        return action
    
    def ground_truth_action(self, last_action):
        action = last_action.copy()
        base_position, _ = p.getBasePositionAndOrientation(self.rodUid)
        action[1] = base_position[1]
        # print("GT action:", action)
        return action
    
    def elevate(self, last_action, lift_distance=0.03):
        action = last_action.copy()
        action[2] += lift_distance
        return action
    
    def lower(self, last_action, lift_distance=0.03):
        action = last_action.copy()
        action[2] -= lift_distance
        return action
    
    def get_image(self):
        import cv2
        obs = self.camera.render(self.physicsClientId)
        rgb = obs["rgb"]
        rgb_img = np.reshape(rgb, (480, 640, 4))
        rgb_img = rgb_img[:, :, :3]
        output_path = "camera_image.png"

        import matplotlib.pyplot as plt
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.show()
        
    def get_flow(self, P_world, pc_seg_obj, action, option="torque", normalize_pcd=False):
        P_world_rod = P_world.copy()
        P_world_rod[pc_seg_obj!=1] = 0
        
        # center_of_mass = self.get_center_of_mass(P_world_rod[pc_seg_obj==1])

        # distance to x
        if normalize_pcd:
            action_in_pcd = action[:3] * self.scale_factors
        else:
            action_in_pcd = action
        x_center_of_mass = [0, action_in_pcd[1], 0]
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
        return flow
    
    def get_score(self):
        rod_y = self.y_offset * 2
        gt_y = self.rodMassPosition[0]
        
        contacts = p.getContactPoints(self.pandaUid, self.rodUid)
        if(len(contacts)==0):
            print("no contact!")
            return None
        predict_y = np.mean(np.array([item[6][1] for item in contacts]))
        # print(predict_y, gt_y, rod_y)
        score = (predict_y-gt_y)/rod_y

        return score
    
    def get_center_of_mass(self, P_world_rod=None):
        # obj size here represents for the ground truth size of the object
        obj_size, center_of_mass_rod = self._get_urdf_shape(self.rodPath)
        # base position here represents for the center of mass ground truth position
        base_position, base_orientation = p.getBasePositionAndOrientation(self.rodUid)
        
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
        
    def normalize_pcd(self, P_world_rod_and_cuboid):
        # print("normalize_pcd", P_world_rod_and_cuboid)
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
        
        return normalized_pc, scale_factors, pcd_center
    
    def update_tracker(self, pc_obs):
        seg = pc_obs['seg']
        segmap = pc_obs['segmap']
        image_seg_id = [None, None, None]
        for id, (obj_id, link_id) in segmap.items():
            if obj_id == self.rodUid:
                image_seg_id[0] = id
            if obj_id == self.supporting_cuboid_1:
                image_seg_id[1] = id
            if obj_id == self.supporting_cuboid_2:
                image_seg_id[2] = id
            
        rgb = pc_obs['rgb'].astype(np.float32)
        depth = pc_obs['depth']
        mask = (seg == image_seg_id[0]) | (seg == image_seg_id[1]) | (seg == image_seg_id[2])
        self.tracker.append_observation(rgb, depth, mask)
        
    def get_latest_obs_flow(self, P_world, normalize_pcd=False):
        if not normalize_pcd:
            return self.tracker.get_latest_obs_flow(P_world)
        else:
            # P_world_org = P_world / self.scale_factors + self.pcd_center
            # print("P_world_org", P_world_org)
            obs_flow = self.tracker.get_latest_obs_flow(P_world)
            # print(obs_flow)
            obs_normalized = (obs_flow) * self.scale_factors
            return obs_normalized
    
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
                        # print(f"  Collision Shape: Box, Size: {size}")
                    elif shape_type == 'mesh':
                        size = geometry.find('mesh').get('scale')

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

def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    eye = np.array([1, 1, 1])
    lookat = np.array([0, 0, 0.2])
    up = np.array([0, 0, 1])
    front = (lookat - eye) / np.linalg.norm(lookat - eye)
    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_lookat(lookat)
    ctr.set_zoom(0.5)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image("pointcloud.png")

    time.sleep(1)
    vis.destroy_window()

def main(): # this is a demo of a single init-lift-lower process
    env = ScoopEnv(rod_id = '3', data_path = "~/datasets/unevenobject/raw/toy")#use_GUI=True)
    # env = ScoopEnv(rod_id = 'knife', data_path = "~/datasets/unevenobject/raw")#use_GUI=True)
    env.seed(42)
    _ = env.init_reset()
    env.init_video_writer("debug.mp4")
    # p.changeVisualShape(env.rodUid, linkIndex=-1, rgbaColor=[1-231/256, 1-215/256, 1-193/256, 1])
    # # visualize and save pointcloud
    # points = render["P_world"]
    # visualize_point_cloud(points)

    last_action = env.prepare_scoop()

    # get the image of initialization
    render = env.render()
    initial_image = render["rgb"]
    cv2.imwrite('debug_initial_image.jpg', initial_image)
    
    print(f'left_most: {env.left_most}')
    print(env.x_offset, env.y_offset, env.z_offset)
    print(f'center of mass: {env.get_center_of_mass()}')

    env.change_collision_with_object(turn_on=True) # Turn on collision check with object after scoop prepared

    step_cnt = 3
    # generate the trajectory
    for i in range(step_cnt):
        # pointcloud
        render = env.render()
        points = render["P_world"]

        # if already lifted, end this trajectory
        lifted = render["lifted"]
        if lifted:
            break

        # if steady, lifted higher
        steady = render["steady"]

        if steady and i != 0:
            randomize = 2
        else:
            if i >= step_cnt - 2:
                # randomize = 1
                randomize = 0
            else:
                # randomize = round(np.random.uniform(0, 1.5))
                randomize = 1


        if i != 0:
            action = env.lower(last_action)
            # action = env.lower(action)
            action = env.lower(action)
            print(f"Action lower: {action}")
            _ = env.step(action, True)

            # print(p.getGravity())  # Should be something like [0, 0, -9.8]
            
            # contacts = p.getContactPoints(env.pandaUid, env.rodUid)
            # print(contacts)
            
            last_action = action

        env.change_collision_with_object(turn_on=False)
        if randomize == 1:
            # action = env.ground_truth_action(last_action)
            # print(f"Action ground truth: {action}")
            
            action = env.choose_action_random(last_action)
            
            print(f"Action random: {action}")
            print(action)
            _ = env.step(action, True)
            print("Shifted!")
            

            last_action = action
        else:
            action = env.ground_truth_action(last_action)
            print(f"Action ground truth: {action}")
            _ = env.step(action, True)
            last_action = action

        env.change_collision_with_object(turn_on=True)
        
        action = env.elevate(last_action)
        action = env.elevate(action)
        # action = env.elevate(action)
        print(f"Action elevate: {action}")
        _ = env.step(action, True)
        
        last_action = action

    render = env.render()
    final_image = render["rgb"]
    cv2.imwrite('debug_final_image.jpg', final_image)
    lifted = render["lifted"]
    steady = render["steady"]
    print(f"Lifted: {lifted}, Stable: {steady}")

    env.close()

if __name__ == "__main__":
    main()