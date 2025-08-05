import torch
import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


class Tracker():
    def __init__(self, camera_intrinsics, T_world_from_cam, crop):
        self.camera_intrinsics = camera_intrinsics
        self.T_world_from_cam = torch.tensor(T_world_from_cam).float()
        self.T_cam_from_world = torch.inverse(self.T_world_from_cam).float()
        
        sys.path.append("/home/yishu/failure_recovery/sequential_predictor/tracking/DELTA_densetrack3d")
        from densetrack3d.models.densetrack3d.densetrack3d import DenseTrack3D
        from densetrack3d.models.predictor.predictor import Predictor3D
        self.model = DenseTrack3D(
            stride=4,
            window_len=16,
            add_space_attn=True,
            num_virtual_tracks=64,
            model_resolution=(crop[1] - crop[0], crop[3] - crop[2]),#(384, 512),
            upsample_factor=2
        )
        ckpt_path = "/home/yishu/failure_recovery/sequential_predictor/tracking/DELTA_densetrack3d/checkpoints/densetrack3d.pth"
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

        self.start_x = crop[0]
        self.end_x = crop[1]

        self.start_y = crop[2]
        self.end_y = crop[3]


    def append_observation(self, rgb, depth):
        self.all_rgbs.append(rgb)
        self.all_depths.append(depth)


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


    def get_inverse_pcd(self, P_world_raw, num_samples=200):
        # print(P_world_raw.shape)
        P_world = P_world_raw[torch.randperm(P_world_raw.shape[0])[:num_samples]]

        # T_cam_from_world = torch.inverse(torch.tensor(self.T_world_from_cam)).float()  # Inverse of world2cam

        # Convert world coordinates to homogeneous coordinates
        # print(P_world.shape,torch.ones((len(P_world), 1)).shape)
        Ph_world = torch.cat([torch.tensor(P_world), torch.ones((len(P_world), 1))], dim=-1)  # (N, 4)

        # Transform to camera coordinates
        Ph_cam = (self.T_cam_from_world @ Ph_world.T).T  # (N, 4)
        P_cam = Ph_cam[:, :3]

        # Camera intrinsics
        K = torch.tensor(self.camera_intrinsics)  # Shape (3,3)

        # Extract X, Y, Z in camera frame
        px_cam, py_cam, pz_cam = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

        # Project to image space
        px = K[0, 0] * (px_cam / pz_cam) + K[0, 2] - self.start_y
        py = K[1, 1] * (py_cam / pz_cam) + K[1, 2] - self.start_x
        pz = pz_cam  # Depth remains unchanged

        # Stack pixel coordinates
        pixel_coords = torch.stack([px, py], dim=-1)
        return pixel_coords, P_world
    
    # def get_inverse_pcd(self, pcd_depth_frame, num_samples=400):
    #     """
    #     Inputs:
    #         - pcd_depth_frame: torch.Tensor of shape (N, 3), point cloud in depth_camera_link
    #         - self.T_rgb_from_depth: 4x4 numpy array, transform depth → rgb
    #         - self.K_rgb: 3x3 numpy array, RGB intrinsics matrix
    #     Output:
    #         - pixel_coords: (num_samples, 2), [u, v] pixel locations in RGB image
    #     """

    #     # Sample subset
    #     pcd_sampled = torch.from_numpy(pcd_depth_frame)[torch.randperm(pcd_depth_frame.shape[0])[:num_samples]]  # (num_samples, 3)

    #     # Homogenize: (N, 3) → (N, 4)
    #     ones = torch.ones((len(pcd_sampled), 1))
    #     Ph_depth = torch.cat([pcd_sampled, ones], dim=-1)  # (N, 4)

    #     # Transform depth → rgb frame
    #     # Ph_rgb = (self.T_depth2rgb @ Ph_depth.T).T  # (N, 4)
    #     # print(Ph_rgb)
    #     Ph_rgb = (self.T_dpeth2rgb @ Ph_depth.T).T
    #     P_rgb = Ph_rgb[:, :3]

    #     # Project to RGB image using K_rgb
    #     K_rgb = torch.tensor(self.camera_intrinsics).float()  # shape (3, 3)

    #     x, y, z = P_rgb[:, 0], P_rgb[:, 1], P_rgb[:, 2]

    #     # Avoid divide-by-zero
    #     z = torch.clamp(z, min=1e-6)

    #     u = K_rgb[0, 0] * (x / z) + K_rgb[0, 2]  # fx * x/z + cx
    #     v = K_rgb[1, 1] * (y / z) + K_rgb[1, 2]  # fy * y/z + cy

    #     pixel_coords = torch.stack([u, v], dim=-1)  # (N, 2)

    #     return pixel_coords



    def get_latest_obs_flow(self, P_world):  
        # data_json = {
        #     "rgbs": self.all_rgbs,
        #     "depths": self.all_depths,
        #     "query_pcd": P_world,
        #     # "obs_flow": dense_track_flows,
        # }
        # import pickle as pkl
        # pkl.dump(data_json, open("new_previous_obs_flow_input.pkl", "wb"))
        # breakpoint()

        start_frame_id=-2
        video = np.stack(self.all_rgbs, axis=0)
        video = torch.from_numpy(video).cuda()[:, :, :, :-1].unsqueeze(0).permute(0, 1, 4, 2, 3)
        # print(video.shape)

        depths = np.stack(self.all_depths, axis=0)
        depths = torch.from_numpy(depths.astype(np.float32)).float().cuda().unsqueeze(1).unsqueeze(0)
        # print(depths.shape)

        # seg_masks = np.stack(self.all_masks, axis=0)
        # seg_masks = torch.from_numpy(seg_masks).float().unsqueeze(0)#.cuda()

        queries_pcd, sampled_pcd = self.get_inverse_pcd(P_world)

        # Visualize the queries on rgb
        img = self.all_rgbs[-2]  # shape: (H, W, 3)
        queries = queries_pcd.numpy()  # shape: (N, 2)

        print("Query min/max x:", queries[:, 0].min(), queries[:, 0].max())
        print("Query min/max y:", queries[:, 1].min(), queries[:, 1].max())
        print("Image size:", img.shape[1], img.shape[0])  # W, H

        plt.imshow(img/256)
        plt.scatter(queries[:, 0], queries[:, 1], c='red', s=10)  # x = col, y = row
        plt.title("2D Query Points on Image")
        plt.axis("off")
        plt.show()


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
        px = pred_tracks[0, start_frame_id, :, 0] + self.start_y
        py = pred_tracks[0, start_frame_id, :, 1] + self.start_x
        pz = pred_tracks[0, start_frame_id, :, 2]
        px = (px - self.camera_intrinsics[0, 2]) * (pz / self.camera_intrinsics[0, 0])
        py = (py - self.camera_intrinsics[1, 2]) * (pz / self.camera_intrinsics[1, 1])
        
        # pcd_ids = pred_tracks[0, :, :, 0] * 640 + pred_tracks[0, :, :, 1]
        orig_pcds = torch.stack([px, py, pz], dim=-1).cpu().numpy()
        Ph_cam = np.concatenate([orig_pcds, np.ones((len(orig_pcds), 1))], axis=1)
        Ph_world = (self.T_world_from_cam @ Ph_cam.T).T
        P_world_orig = Ph_world[:, :3]

        # orig_pcds = P_worlds[0][pcd_ids]
        px = pred_tracks[0, start_frame_id+1, :, 0] + self.start_y
        py = pred_tracks[0, start_frame_id+1, :, 1] + self.start_x
        pz = pred_tracks[0, start_frame_id+1, :, 2]
        px = (px - self.camera_intrinsics[0, 2]) * (pz / self.camera_intrinsics[0, 0])
        py = (py - self.camera_intrinsics[1, 2]) * (pz / self.camera_intrinsics[1, 1])

        new_pcds = torch.stack([px, py, pz], dim=-1).cpu().numpy()
        Ph_cam = np.concatenate([new_pcds, np.ones((len(new_pcds), 1))], axis=1)
        Ph_world = (self.T_world_from_cam @ Ph_cam.T).T
        # Ph_world = Ph_cam
        P_world_new = Ph_world[:, :3]

        sparse_obs_flow = P_world_new - P_world_orig
        # print(np.nan_to_num(sparse_obs_flow, nan=100.0))
        # norms = np.linalg.norm(sparse_obs_flow, axis=1)
        # # Maximum flow is 80% quantile
        # max_flow_norm = np.quantile(norms, 0.2)
        # sparse_obs_flow[norms > max_flow_norm] = sparse_obs_flow[norms > max_flow_norm] / norms[norms > max_flow_norm][:, np.newaxis] * max_flow_norm

        # return P_world_orig, sparse_obs_flow#dense_track_flows

        # Interpolate to get points for all of the P_world points
        dense_track_flows = griddata(P_world_orig, sparse_obs_flow, P_world, method='linear')
        # print(np.nan_to_num(dense_track_flows, nan=100.0))
        dense_track_flows = np.nan_to_num(dense_track_flows, nan=0.0)

        # Save the current dataset
        # data_json = {
        #     "rgbs": self.all_rgbs,
        #     "depths": self.all_depths,
        #     "query_pcd": P_world,
        #     "obs_flow": dense_track_flows,
        # }
        # import pickle as pkl
        # pkl.dump(data_json, open("new_latest_obs_flow_input.pkl", "wb"))
        # breakpoint()
        return dense_track_flows, sampled_pcd, P_world_orig, P_world_new, sparse_obs_flow, pred_tracks