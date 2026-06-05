import copy
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose
from mmpose.utils.visualization.skeleton import Skeleton
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from mmpose.core.evaluation.top_down_eval import pose_pck_accuracy, _get_max_preds, _get_softargmax_preds
from mmpose.datasets.datasets.egocentric.joint_converter import dset_to_body_model
import json
from ...builder import DATASETS
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated



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
    """
    volume: [D, H, W]
    center: (z, y, x) in voxel float coordinates
    """
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
            continue

        dist = torch.norm(p2 - p1).item()
        n_points = max(int(dist * 2), 1)


        for t in torch.linspace(0, 1, steps=n_points):
            pt = p1 * (1 - t) + p2 * t
            draw_gaussian_3d(heatmaps[idx], pt, sigma)
            
            draw_count += 1

        max_val = heatmaps[idx].max()
        nonzero_count = (heatmaps[idx] > 1e-6).sum().item()

    return heatmaps




@DATASETS.register_module()
class LimbMocapStudioDataset(Dataset):   

    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe', 'pck']

    path_dict = {
        'new_jian1': {
            'path': r'path to /SceneEgo/test/jian1',
        },                                                                  
        'new_jian2': {
            'path': r'path to /SceneEgo/test/jian2',
        },                                                                  
        'new_diogo1': {
            'path': r'path to /SceneEgo/test/diogo1',
        },                                                                  
        'new_diogo2': {
            'path': r'path to /SceneEgo/test/diogo2',
        },                                                                  

    }

    def __init__(self,
                 data_cfg,
                 pipeline,
                 local=False,
                 test_mode=False,
                 sample_a_few=False,
                 ):

        self.data_cfg = copy.deepcopy(data_cfg)
        self.local = local
        if local:
            for key in self.path_dict.keys():
                self.path_dict[key]['path'] = self.path_dict[key]['path'].replace('/HPS', 'X:')
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.ann_info = {}

        self.load_config(self.data_cfg)
        self.skeleton = Skeleton(self.ann_info['camera_param_path'])

        self.dst_idxs, self.model_idxs = dset_to_body_model(dset='mo2cap2', model_type=self.ann_info['joint_type'],
                use_face_contour=False)

        self.data_info = self.load_annotations()
        self.pipeline = Compose(pipeline)

        if sample_a_few:
            print('!!!!!warning, only sample a few data for debug!!!!!!!')
            self.data_info = self.data_info[::100]


    def load_config(self, data_cfg):
        """Initialize dataset attributes according to the config.

        Override this method to set dataset specific attributes.
        """
        self.ann_info = copy.deepcopy(data_cfg)
        self.ann_info['image_size'] = np.asarray(self.ann_info['image_size'])


    def load_annotations(self):
        """Load data annotation."""
        print("start loading test file")
        data_info = []
        for seq_name in self.path_dict:
            base_path = self.path_dict[seq_name]['path']

            img_data_path = os.path.join(base_path, 'imgs')
            gt_path = os.path.join(base_path, 'local_pose_gt.pkl')
            syn_path = os.path.join(base_path, 'syn.json')

            with open(syn_path, 'r') as f:
                syn_data = json.load(f)

            ego_start_frame = syn_data['ego']
            ext_start_frame = syn_data['ext']

            with open(gt_path, 'rb') as f:
                pose_gt_data = pickle.load(f)

            image_path_list = []
            gt_pose_list = []

            for pose_gt_item in pose_gt_data[:500]:                   
                ext_id = pose_gt_item['ext_id']
                keypoints_3d = pose_gt_item['ego_pose_gt']
                calib_board_pose = pose_gt_item['calib_board_pose']
                if keypoints_3d is None:
                    print(f"None pose in base_path: {base_path} and ext_id: {ext_id}")
                    continue
                ego_id = ext_id - ext_start_frame + ego_start_frame
                if ext_id < ext_start_frame:
                    continue

                ego_id = ego_id + (ext_id - ext_start_frame) // 1000
                if (ext_id - ext_start_frame) % 1000 > 1000 / 2:
                    # print('warning: correct unsynchronized data')
                    ego_id += 1

                egocentric_image_name = "img_%06d.jpg" % ego_id

                image_path = os.path.join(img_data_path, egocentric_image_name)
                if not os.path.exists(image_path):
                    continue
                image_path_list.append(image_path)

                keypoints_3d_visible = np.ones([15], dtype=np.float32)

                gt_pose_list.append(keypoints_3d)
                data_info.append(
                    {
                        'seq_name': seq_name,
                        'ext_id': ext_id,
                        'image_file': image_path,
                        'keypoints_3d': keypoints_3d,
                        'calib_board_pose': calib_board_pose,
                        'keypoints_3d_visible': keypoints_3d_visible
                    }
                )
        return data_info

    def evaluate(self, outputs, res_folder, metric=['mpjpe', 'pa-mpjpe', 'pck'], logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise KeyError(f'metric {metric} is not supported, supported metrics are {self.allowed_metrics}')

        evaluation_results = {}
        for metric_name in metrics:
            evaluation_numbers = self._report_metric_and_save(outputs, metric_name, res_folder)
            evaluation_results = {**evaluation_results, **evaluation_numbers}

            
        return evaluation_results


    def _report_metric_and_save(self, result_list, metric_name, res_folder):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """
        eval_res_dict = {}
        if 'mpjpe' in metric_name or 'pa-mpjpe' in metric_name or 'ba-mpjpe' in metric_name:
            pred_joints_3d_list = []
            gt_joints_3d_list = []
            gt_joints_visible_list = []
            for pred in result_list:
                pred_joints_3d_item = pred['keypoints_pred']

                if torch.is_tensor(pred_joints_3d_item):
                    pred_joints_3d_item = pred_joints_3d_item.cpu().numpy()

                pred_joints_3d_item = self._convert_from_model_format_to_mo2cap2_format(pred_joints_3d_item,
                                                                  model_idxs=self.model_idxs,
                                                                  dst_idxs=self.dst_idxs)
                pred_joints_3d_list.extend(pred_joints_3d_item)
                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    gt_joints_3d_item = img_meta_item['keypoints_3d']
                    gt_joints_3d_list.append(gt_joints_3d_item)
                    gt_joints_visible_list.append(img_meta_item['keypoints_3d_visible'])

            pred_joints_3d = np.array(pred_joints_3d_list)
            gt_joints_3d = np.array(gt_joints_3d_list)
            gt_joints_visible = np.array(gt_joints_visible_list).astype(bool)

            assert len(pred_joints_3d[0]) == len(gt_joints_3d[0])
            if 'mpjpe' in metric_name:
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res_mpjpe = keypoint_mpjpe(pred_joints_3d_copy, gt_joints_3d_copy, gt_joints_visible, alignment='none')
                eval_res_dict['mpjpe'] = eval_res_mpjpe

                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res_pa_mpjpe = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='procrustes')
                eval_res_dict['pa-mpjpe'] = eval_res_pa_mpjpe

            if 'pa-mpjpe' in metric_name:
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='procrustes')
                eval_res_dict['pa-mpjpe'] = eval_res
            if 'ba-mpjpe' in metric_name:
                pred_joints_3d_copy = copy.deepcopy(pred_joints_3d)
                gt_joints_3d_copy = copy.deepcopy(gt_joints_3d)
                eval_res = keypoint_mpjpe(
                    pred_joints_3d_copy,
                    gt_joints_3d_copy,
                    gt_joints_visible,
                    alignment='bone_length')
                eval_res_dict['ba-mpjpe'] = eval_res

            if res_folder is not None:
                print(f'save to {res_folder}')
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump(result_list, f)

        if 'pck' in metric_name:
            heatmaps_2d_pred = []
            heatmaps_2d_pred_original = []
            heatmaps_2d_gt = []
            image_file_list = []

            for pred in result_list:
                output_heatmap_item = pred['output_heatmap']
                heatmaps_2d_pred_original.extend(output_heatmap_item)

                output_heatmap_item = self._convert_from_model_format_to_mo2cap2_format(output_heatmap_item,
                                                                                        model_idxs=self.model_idxs,
                                                                                        dst_idxs=self.dst_idxs)
                heatmaps_2d_pred.extend(output_heatmap_item)
                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    target = img_meta_item['target']
                    heatmaps_2d_gt.append(target)

                    image_file_list.append(img_meta_item['image_file'])
            heatmaps_2d_pred = np.asarray(heatmaps_2d_pred)
            heatmaps_2d_pred_original = np.asarray(heatmaps_2d_pred_original)
            heatmaps_2d_gt = np.asarray(heatmaps_2d_gt)

            N, K, H, W = heatmaps_2d_pred.shape
            mask = np.ones((N, K)).astype(np.bool)
            acc, avg_acc, cnt = pose_pck_accuracy(heatmaps_2d_pred, heatmaps_2d_gt, mask, thr=0.05)
            eval_res_dict['pck'] = avg_acc
            eval_res_dict['joint_pck'] = acc

            if res_folder is not None:
                print(f'save to {res_folder}')

                joints_2d_pred, _ = _get_softargmax_preds(heatmaps_2d_pred_original)
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump({'joints_2d_pred': joints_2d_pred, 'image_file': image_file_list}, f)

        if len(eval_res_dict) == 0:
            raise KeyError(f'metric {metric_name} is not supported, supported metrics are {self.allowed_metrics}')
        return eval_res_dict

    def _convert_from_model_format_to_mo2cap2_format(self, pred_joints, model_idxs, dst_idxs):
        mo2cap2_shape = list(pred_joints.shape)
        mo2cap2_shape[1] = 15
        mo2cap2_joint_batch = np.empty(mo2cap2_shape)

        mo2cap2_joint_batch[:, dst_idxs] = pred_joints[:, model_idxs]
        return mo2cap2_joint_batch



    def prepare_data(self, idx):
        """Get data sample."""
        result = self.data_info[idx]

        if 'keypoints_3d' in result:
            keypoints_3d = np.asarray(result['keypoints_3d']).astype(np.float32)

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

            result['target_limb_heatmap'] = limb_gt

        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        # print(f'get item {idx}')
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
