import numpy as np
import os
import torch
import sys
from scipy.interpolate import griddata

class Tracker():
    def __init__(self, camera, delta_path=None, delta_ckpt_path=None):
        self.camera = camera
        if delta_path is not None and delta_path not in sys.path:
            sys.path.insert(0, delta_path)
            
        from have.utils.densetrack3d import DenseTrack3D
        from have.utils.DELTA.densetrack3d.models.predictor.predictor import Predictor3D
        self.model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(480, 640),#(384, 512),
            upsample_factor=2
        )
        ckpt_path = delta_ckpt_path
        with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.predictor = Predictor3D(model=self.model)
        self.predictor = self.predictor.eval().cuda()

        self.all_rgbs = []
        self.all_depths = []
        self.all_masks = []


    def append_observation(self, rgb, depth, mask):
        self.all_rgbs.append(rgb)
        self.all_depths.append(depth)
        self.all_masks.append(mask)


    def get_closest_object_point(self, query_x, query_y, depth, object_mask, intrinsics):
        # Find the closest (x, y) point that has 1 in object mask, consist their x, y and depth into a point cloud.
        object_points = np.argwhere(object_mask == 1)
        distances = np.sum((object_points - np.array([query_x.cpu().numpy(), query_y.cpu().numpy()]))**2, axis=1)
        
        # Find the closest point
        closest_idx = np.argmin(distances)
        if distances[closest_idx] > 3:
            return None
        closest_x, closest_y = object_points[closest_idx]
        this_depth = depth[closest_x, closest_y]

        px = (query_y - intrinsics[0, 2]) * (this_depth / intrinsics[0, 0])
        py = (query_x - intrinsics[1, 2]) * (this_depth / intrinsics[1, 1])
        
        # Get depth at closest point
        return np.float32([px.item(), py.item(), this_depth])

    def get_inverse_pcd(self, P_world_raw, num_samples=400):
        # print(P_world_raw.shape)
        P_world = P_world_raw[torch.randperm(P_world_raw.shape[0])[:num_samples]]

        T_cam2world = torch.inverse(torch.tensor(self.camera.T_world2cam))  # Inverse of world2cam

        # Convert world coordinates to homogeneous coordinates
        # print(P_world.shape,torch.ones((len(P_world), 1)).shape)
        Ph_world = torch.cat([torch.tensor(P_world), torch.ones((len(P_world), 1))], dim=-1)  # (N, 4)

        # Transform to camera coordinates
        Ph_cam = (T_cam2world @ Ph_world.T).T  # (N, 4)
        P_cam = Ph_cam[:, :3]

        # Camera intrinsics
        K = torch.tensor(self.camera.intrinsics)  # Shape (3,3)

        # Extract X, Y, Z in camera frame
        px_cam, py_cam, pz_cam = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

        # Project to image space
        px = K[0, 0] * (px_cam / pz_cam) + K[0, 2]
        py = K[1, 1] * (py_cam / pz_cam) + K[1, 2]
        pz = pz_cam  # Depth remains unchanged

        # Stack pixel coordinates
        pixel_coords = torch.stack([px, py], dim=-1)
        return pixel_coords


    def get_latest_obs_flow(self, P_world):  
        start_frame_id=-2
        video = np.stack(self.all_rgbs, axis=0)
        video = torch.from_numpy(video).cuda().unsqueeze(0).permute(0, 1, 4, 2, 3) # [:, :, :, :-1]
        # print(video.shape)

        depths = np.stack(self.all_depths, axis=0)
        depths = torch.from_numpy(depths).float().cuda().unsqueeze(1).unsqueeze(0)
        # print(depths.shape)

        seg_masks = np.stack(self.all_masks, axis=0)
        seg_masks = torch.from_numpy(seg_masks).float().unsqueeze(0)#.cuda()

        queries_pcd = self.get_inverse_pcd(P_world)
        queries_pcd = torch.concatenate([torch.ones((queries_pcd.shape[0], 1)) * (len(self.all_rgbs)-2), queries_pcd], dim=-1)

        # Inference with DELTA
        out_dict = self.predictor(
                video,
                depths,
                queries=queries_pcd.unsqueeze(0).float().to(video.device),
                segm_mask=None,
                grid_size=20,
                grid_query_frame=0,
                backward_tracking=False,
                predefined_intrs=None
            )
        

        trajs_3d_dict = {k: v[0].cpu().numpy() for k, v in out_dict["trajs_3d_dict"].items()}
        # return trajs_3d_dict, out_dict
        
        # msk_query = (T_Firsts == 0)

        pred_tracks = torch.concat([out_dict['trajs_uv'], out_dict['trajs_depth']], dim=-1) # pred_tracks[:,:,msk_query.squeeze()]
        px = pred_tracks[0, start_frame_id, :, 0]
        py = pred_tracks[0, start_frame_id, :, 1]
        pz = pred_tracks[0, start_frame_id, :, 2]
        px = (px - self.camera.intrinsics[0, 2]) * (pz / self.camera.intrinsics[0, 0])
        py = (py - self.camera.intrinsics[1, 2]) * (pz / self.camera.intrinsics[1, 1])
        
        # pcd_ids = pred_tracks[0, :, :, 0] * 640 + pred_tracks[0, :, :, 1]
        orig_pcds = torch.stack([px, py, pz], dim=-1).cpu().numpy()
        Ph_cam = np.concatenate([orig_pcds, np.ones((len(orig_pcds), 1))], axis=1)
        Ph_world = (self.camera.T_world2cam @ Ph_cam.T).T
        P_world_orig = Ph_world[:, :3]

        # orig_pcds = P_worlds[0][pcd_ids]
        px = pred_tracks[0, start_frame_id+1, :, 0]
        py = pred_tracks[0, start_frame_id+1, :, 1]
        pz = pred_tracks[0, start_frame_id+1, :, 2]
        px = (px - self.camera.intrinsics[0, 2]) * (pz / self.camera.intrinsics[0, 0])
        py = (py - self.camera.intrinsics[1, 2]) * (pz / self.camera.intrinsics[1, 1])

        new_pcds = torch.stack([px, py, pz], dim=-1).cpu().numpy()
        Ph_cam = np.concatenate([new_pcds, np.ones((len(new_pcds), 1))], axis=1)
        Ph_world = (self.camera.T_world2cam @ Ph_cam.T).T
        P_world_new = Ph_world[:, :3]

        sparse_obs_flow = P_world_new - P_world_orig

        # return P_world_orig, sparse_obs_flow#dense_track_flows

        # Interpolate to get points for all of the P_world points
        dense_track_flows = griddata(P_world_orig, sparse_obs_flow, P_world, method='linear')
        dense_track_flows = np.nan_to_num(dense_track_flows, nan=0.0)
        return dense_track_flows