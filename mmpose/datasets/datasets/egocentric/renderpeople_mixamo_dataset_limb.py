import copy
import os
import pickle
import torch

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated


from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from .joint_converter import dset_to_body_model
from ...builder import DATASETS



LIMB_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14), (7, 11)
]

def fisheye2voxel(xy_2d: torch.Tensor, depth: torch.Tensor,
                  voxel_size=(2, 2, 2), heatmap_size=(64, 64, 64)) -> torch.Tensor:
    assert xy_2d.shape[1] == 2
    assert xy_2d.shape[0] == depth.shape[0]

    device = xy_2d.device
    J = xy_2d.shape[0]

    x = xy_2d[:, 0]
    y = xy_2d[:, 1]
    z = depth

    W, H, D = heatmap_size[2], heatmap_size[1], heatmap_size[0]

    voxel_x = (x - 128.0) / 1024.0 * W
    voxel_y = y / 1024.0 * H
    voxel_z = z / voxel_size[2] * D

    joints_voxel = torch.stack([voxel_z, voxel_y, voxel_x], dim=1)  

    return joints_voxel


def draw_gaussian_3d(volume, center, sigma):

    D, H, W = volume.shape
    z, y, x = center.tolist()

    radius = int(3 * sigma + 0.5)
    z_min, z_max = int(z) - radius, int(z) + radius + 1
    y_min, y_max = int(y) - radius, int(y) + radius + 1
    x_min, x_max = int(x) - radius, int(x) + radius + 1

    zz, yy, xx = torch.meshgrid(
        torch.arange(z_min, z_max),
        torch.arange(y_min, y_max),
        torch.arange(x_min, x_max),
        indexing='ij'
    )

    mask = (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    zz, yy, xx = zz[mask], yy[mask], xx[mask]

    dz = zz.float() - z
    dy = yy.float() - y
    dx = xx.float() - x
    g = torch.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))



    volume[zz, yy, xx] += g

def generate_limb_heatmap_3d(joints_3d, heatmap_size=(64, 64, 64), sigma=1.5):      
    num_limbs = len(LIMB_CONNECTIONS)
    D, H, W = heatmap_size
    heatmaps = torch.zeros((num_limbs, D, H, W), dtype=torch.float32)     

    draw_count = 0


    for idx, (j1, j2) in enumerate(LIMB_CONNECTIONS):
        p1 = joints_3d[j1]
        p2 = joints_3d[j2]

        if torch.any(torch.isnan(p1)) or torch.any(torch.isnan(p2)):
            print(f"limb {idx} skipped due to NaN")
            continue

        dist = torch.norm(p2 - p1).item()
        n_points = max(int(dist * 2), 1)


        for t in torch.linspace(0, 1, steps=n_points):
            pt = p1 * (1 - t) + p2 * t
            draw_gaussian_3d(heatmaps[idx], pt, sigma)
            
        max_val = heatmaps[idx].max()
        nonzero_count = (heatmaps[idx] > 1e-6).sum().item()
        #print(f"Limb {idx}: nonzero voxels = {nonzero_count}")


    return heatmaps

@DATASETS.register_module()
class LimbRenderpeopleMixamoDataset(Dataset):        
    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe']

    def __init__(self,
                 ann_file,      
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dset='renderpeople_old',
                 test_mode=False):

        self.ann_file = ann_file                        
        self.img_prefix = img_prefix                    
        self.data_cfg = copy.deepcopy(data_cfg)        
        self.pipeline = pipeline                        
        self.test_mode = test_mode                      
        self.dset = dset
        self.ann_info = {}

        self.load_config(self.data_cfg)         
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)



        if self.ann_info['joint_type'] == 'smplx':
            self.dst_idxs, self.model_idxs = dset_to_body_model(
                dset=dset,
                model_type='smplx',
                use_face_contour=False)

    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])

    def load_annotations(self):
        """Load data annotation."""
        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)

        if self.dset == 'renderpeople_old':
            data_list = data['data_list']
        elif self.dset == 'renderpeople':
            data_list = []
            data_raw_list = data['data_list']
            for identity_name, identity_data in data_raw_list.items():
                print(f'load identity: {identity_name}')
                for seq_name, seq_data in tqdm(identity_data.items()):
                    data_list.extend(seq_data)
        else:
            raise Exception('dset type is incorrect')

        if self.test_mode is True:
            data_list = data_list[:200]

        
        return data_list

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe'], logger=None):
        res_file = os.path.join(res_folder, 'results.pkl')
        with open(res_file, 'wb') as f:
            pickle.dump(outputs, f)

        return {'result': 0}

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'wb') as f:
            pickle.dump(keypoints, f)

    def _report_metric(self, res_file, metric_name):
        with open(res_file, 'rb') as fin:
            preds = pickle.load(fin)
        assert len(preds) == len(self.data_info)

        pred_joints_3d = [pred['keypoints'] for pred in preds]
        gt_joints_3d = [item['joints_3d'] for item in self.data_info]

        pred_joints_3d = np.array(pred_joints_3d)
        gt_joints_3d = np.array(gt_joints_3d)

        assert len(pred_joints_3d[0]) == len(gt_joints_3d[0])

        if metric_name == 'mpjpe':
            eval_res = keypoint_mpjpe(pred_joints_3d, gt_joints_3d, alignment='none')
        elif metric_name == 'pa-mpjpe':
            eval_res = keypoint_mpjpe(
                pred_joints_3d,
                gt_joints_3d,
                alignment='procrustes')
        elif metric_name == 'ba-mpjpe':
            eval_res = keypoint_mpjpe(
                pred_joints_3d,
                gt_joints_3d,
                alignment='bone_length')
        else:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        eval_res = {metric_name: eval_res}
        return eval_res

    def prepare_data(self, idx):
        result = {}
        data = self.data_info[idx]
        result['image_file'] = os.path.join(self.img_prefix, data['img_path'])
        result['depth_file'] = os.path.join(self.img_prefix, data['depth_path'])
        result['seg_file'] = os.path.join(self.img_prefix, data['seg_path'])
        if self.ann_info['joint_type'] == 'mo2cap2':
            result['keypoints_3d'] = data['mo2cap2_local_joints']
            N_joints, _ = data['mo2cap2_local_joints'].shape
            result['keypoints_3d_visible'] = np.ones([N_joints], dtype=np.float32)
        elif self.ann_info['joint_type'] == 'renderpeople':
            keypoints_3d = data['renderpeople_local_joints']
            N_joints, _ = keypoints_3d.shape
            keypoints_3d_visible = np.ones([N_joints], dtype=np.float32)

            left_hand_keypoints_3d = np.concatenate([keypoints_3d[22:23], keypoints_3d[33:33 + 20]], axis=0)
            right_hand_keypoints_3d = np.concatenate([keypoints_3d[23:24], keypoints_3d[53:53 + 20]], axis=0)
            left_hand_keypoints_3d_visible = np.concatenate([keypoints_3d_visible[22:23],
                                                             keypoints_3d_visible[33:33 + 20]], axis=0)
            right_hand_keypoints_3d_visible = np.concatenate([keypoints_3d_visible[23:24],
                                                              keypoints_3d_visible[53:53 + 20]], axis=0)

            if 'only_hand' in self.ann_info.keys() and self.ann_info['only_hand'] is True:
                keypoints_3d = np.concatenate([left_hand_keypoints_3d, right_hand_keypoints_3d], axis=0)
                keypoints_3d_visible = np.concatenate([left_hand_keypoints_3d_visible,
                                                       right_hand_keypoints_3d_visible], axis=0)

            result['keypoints_3d'] = keypoints_3d
            result['keypoints_3d_visible'] = keypoints_3d_visible

        elif self.ann_info['joint_type'] == 'smplx':
            keypoints3d = np.zeros([127, 3], dtype=np.float32)
            keypoints3d_visible = np.zeros([127], dtype=np.float32)
            keypoints = data['renderpeople_local_joints']

            keypoints3d[self.model_idxs] = keypoints[self.dst_idxs]
            keypoints3d_visible[self.model_idxs] = 1
            result['keypoints_3d'] = keypoints3d
            result['keypoints_3d_visible'] = keypoints3d_visible


        result['body_pose'] = np.zeros((21 * 3), dtype=np.float32)
        result['betas'] = np.zeros((10,), dtype=np.float32)

        result['has_smpl'] = 0


        if 'keypoints_3d' in result:
            keypoints_3d = result['keypoints_3d']  

            keypoints_3d = np.asarray(keypoints_3d).astype(np.float32)


            heatmap_size = tuple(self.ann_info.get('heatmap_size', (64, 64, 64)))
            if len(heatmap_size) == 2:
                heatmap_size = (heatmap_size[0], heatmap_size[1], heatmap_size[1])
            voxel_size = tuple(self.ann_info.get('voxel_size', (2, 2, 2)))


            
            camera = FishEyeCameraCalibrated(self.ann_info['camera_param_path'])


            keypoints_2d = camera.world2camera(keypoints_3d) 


            if keypoints_2d.ndim == 1:
                keypoints_2d = keypoints_2d.reshape(-1, 2)

            depth_values = keypoints_3d[:, 2]  
            
            joints_voxel = fisheye2voxel(
                torch.from_numpy(keypoints_2d).float(),
                torch.from_numpy(depth_values).float(),
                voxel_size=voxel_size,
                heatmap_size=heatmap_size
            ).numpy()  


            joints_voxel = np.clip(joints_voxel, 0, np.array(heatmap_size) - 1)

            limb_gt = generate_limb_heatmap_3d(
                torch.from_numpy(joints_voxel).float(),
                heatmap_size=heatmap_size,
                sigma=1
                
            )
            torch.set_printoptions(precision=5, sci_mode=False)

            result['target_limb_heatmap'] = limb_gt  



        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info

        return self.pipeline(results)
        
