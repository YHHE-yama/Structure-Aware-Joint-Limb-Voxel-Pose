# Copyright (c) OpenMMLab. All rights reserved.
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
    """
    将图像空间的 2D 坐标和深度值转换为 voxel 空间中的 3D 坐标。

    Args:
        xy_2d: [J, 2] torch.Tensor，图像坐标 (x, y)，单位像素
        depth: [J] torch.Tensor，深度值，
        voxel_size: tuple (vx, vy, vz)，每个 voxel 的实际尺寸，
        heatmap_size: tuple (D, H, W)，输出 voxel 热图的空间尺寸

    Returns:
        joints_voxel: [J, 3] torch.Tensor，(z, y, x) 坐标，浮点 voxel 空间坐标
    """
    assert xy_2d.shape[1] == 2
    assert xy_2d.shape[0] == depth.shape[0]

    device = xy_2d.device
    J = xy_2d.shape[0]

    # === Step 1: 将图像坐标缩放为网络 soft-argmax 输出时的坐标系统 ===
    # 与 Heatmap3DNet 中相反
    x = xy_2d[:, 0]
    y = xy_2d[:, 1]
    z = depth

    # 网络中是：
    #   joint_coord[:, :, 0] = joint_coord[:, :, 0] / W * 1024 + 128
    #   joint_coord[:, :, 1] = joint_coord[:, :, 1] / H * 1024
    #   joint_coord[:, :, 2] = joint_coord[:, :, 2] * (voxel_size[2] / D)

    # 所以反变换如下：
    W, H, D = heatmap_size[2], heatmap_size[1], heatmap_size[0]

    voxel_x = (x - 128.0) / 1024.0 * W
    voxel_y = y / 1024.0 * H
    voxel_z = z / voxel_size[2] * D

    joints_voxel = torch.stack([voxel_z, voxel_y, voxel_x], dim=1)  # 注意 z, y, x 顺序
    #print("=== joints_voxel 检查 ===")
    #print("shape:", joints_voxel.shape)
    #print("min:", joints_voxel.min(), "max:", joints_voxel.max())
    #print("是否包含 NaN:", np.isnan(joints_voxel).any())

    return joints_voxel

'''
def draw_anisotropic_gaussian_3d(volume, center, sigma):
    """
    center: (z, y, x)
    sigma: [σ_z, σ_y, σ_x]
    """
    D, H, W = volume.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij'
    )

    zz = zz.to(torch.float32)
    yy = yy.to(torch.float32)
    xx = xx.to(torch.float32)

    dz = ((zz - center[0]) / sigma[0]) ** 2
    dy = ((yy - center[1]) / sigma[1]) ** 2
    dx = ((xx - center[2]) / sigma[2]) ** 2

    g = torch.exp(-0.5 * (dz + dy + dx))
    volume += g  # or torch.maximum(volume, g)

'''

#'''
#最初的版本
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
    #print()

    # 边界裁剪
    mask = (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    zz, yy, xx = zz[mask], yy[mask], xx[mask]

    dz = zz.float() - z
    dy = yy.float() - y
    dx = xx.float() - x
    g = torch.exp(-(dx**2 + dy**2 + dz**2) / (2 * sigma**2))

    #print(f"▶️ zz.shape = {zz.shape}, g.max = {g.max()}, g.sum = {g.sum()}")

    #if g.sum() > 0:
        #print("✅ 有写入高斯值")
    #else:
        #print("❌ g 全是 0，没写入任何值")


    volume[zz, yy, xx] += g
    #print(f"中心坐标：{center.tolist()}, 体素范围: {volume.shape}")
#'''

def generate_limb_heatmap_3d(joints_3d, heatmap_size=(64, 64, 64), sigma=1.5):      #sigma=[2.0, 0.6, 0.6]
    num_limbs = len(LIMB_CONNECTIONS)
    D, H, W = heatmap_size
    heatmaps = torch.zeros((num_limbs, D, H, W), dtype=torch.float32)
    #print(f"draw_gaussian_3d() 执行次数统计")
    draw_count = 0


    for idx, (j1, j2) in enumerate(LIMB_CONNECTIONS):
        p1 = joints_3d[j1]
        p2 = joints_3d[j2]

        if torch.any(torch.isnan(p1)) or torch.any(torch.isnan(p2)):
            #print(f"⚠️ limb {idx} skipped due to NaN")
            continue

        dist = torch.norm(p2 - p1).item()
        n_points = max(int(dist * 2), 1)
        #print(f"Limb {idx}: dist = {dist:.2f}, n_points = {n_points}")

        for t in torch.linspace(0, 1, steps=n_points):
            pt = p1 * (1 - t) + p2 * t
            #print(f"   ↪ draw at pt: {pt.tolist()}")
            #draw_anisotropic_gaussian_3d(heatmaps[idx], pt, sigma)
            draw_gaussian_3d(heatmaps[idx], pt, sigma)
            
            draw_count += 1
        #print(f"✅ 总共执行 draw_gaussian_3d 次数: {draw_count}")
        #print(f"   → heatmap max: {heatmaps[idx].max().item()}")

        max_val = heatmaps[idx].max()
        nonzero_count = (heatmaps[idx] > 1e-6).sum().item()
        #print(f"Limb {idx}: nonzero voxels = {nonzero_count}")

        ###################测试，尝试注释了这一行
        #if max_val > 0:
        #    heatmaps[idx] /= max_val
            #print(f"Limb {idx}: heatmap max = {max_val}")

    return heatmaps




@DATASETS.register_module()
class LimbMocapStudioDataset(Dataset):   #从多个 egocentric mocap 测试序列中读取图像与对应的标注关键点 3D 姿态，支持灵活的坐标转换（Mo2Cap2 → SMPLX），并实现了全套精度评估流程

    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe', 'pck']

    path_dict = {
        'new_jian1': {
            'path': r'/misc/Corpus3/Human-Pose-Estimation-dataset/data_set/SceneEgo_new/test/jian1',
        },                                                                  #'/HPS/ScanNet/work/egocentric_view/25082022/jian1'
        'new_jian2': {
            'path': r'/misc/Corpus3/Human-Pose-Estimation-dataset/data_set/SceneEgo_new/test/jian2',
        },                                                                  #'/HPS/ScanNet/work/egocentric_view/25082022/jian2'
        'new_diogo1': {
            'path': r'/misc/Corpus3/Human-Pose-Estimation-dataset/data_set/SceneEgo_new/test/diogo1',
        },                                                                  #r'/HPS/ScanNet/work/egocentric_view/25082022/diogo1'
        'new_diogo2': {
            'path': r'/misc/Corpus3/Human-Pose-Estimation-dataset/data_set/SceneEgo_new/test/diogo2',
        },                                                                  #'/HPS/ScanNet/work/egocentric_view/25082022/diogo2'

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

            for pose_gt_item in pose_gt_data[:500]:                   #pose_gt_data[:100]可以只读取100
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
                # convert the mo2cap2 joint representation to smplx joint representation
                # if self.ann_info['joint_type'] == 'smplx':
                #     keypoints_3d_smplx = np.zeros([127, 3], dtype=np.float32)
                #     keypoints_3d_visible = np.zeros([127], dtype=np.float32)
                #     keypoints_3d_smplx[self.model_idxs] = keypoints_3d[self.dst_idxs]
                #     keypoints_3d_visible[self.model_idxs] = 1
                #     keypoints_3d = keypoints_3d_smplx
                # elif self.ann_info['joint_type'] == 'renderpeople':
                #     keypoints_3d_smplx = np.zeros([55, 3], dtype=np.float32)
                #     keypoints_3d_visible = np.zeros([55], dtype=np.float32)
                #     keypoints_3d_smplx[self.model_idxs] = keypoints_3d[self.dst_idxs]
                #     keypoints_3d_visible[self.model_idxs] = 1
                #     keypoints_3d = keypoints_3d_smplx
                # else:
                #     keypoints_3d_visible = np.ones([15, 1], dtype=np.float32)

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

        # res_file = os.path.join(res_folder, 'results.pkl')
        # with open(res_file, 'wb') as f:
        #     pickle.dump(outputs, f)
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

                # convert from model format to mo2cap2 format
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

                # now we estimate both!
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

            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                with open(os.path.join(res_folder, 'results.pkl'), 'wb') as f:
                    pickle.dump(result_list, f)

        if 'pck' in metric_name:
            heatmaps_2d_pred = []
            heatmaps_2d_pred_original = []
            heatmaps_2d_gt = []
            image_file_list = []
            # gt_joints_visible_list = []
            for pred in result_list:
                output_heatmap_item = pred['output_heatmap']
                heatmaps_2d_pred_original.extend(output_heatmap_item)
                # convert from model format to mo2cap2 format
                output_heatmap_item = self._convert_from_model_format_to_mo2cap2_format(output_heatmap_item,
                                                                                        model_idxs=self.model_idxs,
                                                                                        dst_idxs=self.dst_idxs)
                heatmaps_2d_pred.extend(output_heatmap_item)
                img_meta_list = pred['img_metas']
                for img_meta_item in img_meta_list:
                    target = img_meta_item['target']
                    heatmaps_2d_gt.append(target)
                    # gt_joints_visible_list.append(img_meta_item['keypoints_2d_visible'])
                    image_file_list.append(img_meta_item['image_file'])
            heatmaps_2d_pred = np.asarray(heatmaps_2d_pred)
            heatmaps_2d_pred_original = np.asarray(heatmaps_2d_pred_original)
            heatmaps_2d_gt = np.asarray(heatmaps_2d_gt)
            # gt_joints_visible = np.asarray(gt_joints_visible_list).astype(np.bool)
            #print(heatmaps_2d_pred.shape)
            #print(heatmaps_2d_gt.shape)
            N, K, H, W = heatmaps_2d_pred.shape
            mask = np.ones((N, K)).astype(np.bool)
            acc, avg_acc, cnt = pose_pck_accuracy(heatmaps_2d_pred, heatmaps_2d_gt, mask, thr=0.05)
            eval_res_dict['pck'] = avg_acc
            eval_res_dict['joint_pck'] = acc
            # save result to res folder
            if res_folder is not None:
                print(f'save to {res_folder}')
                # joints_2d_pred, _ = _get_max_preds(heatmaps_2d_pred_original)
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

            # 1. 加载 fisheye 相机模型
            camera = FishEyeCameraCalibrated(self.ann_info['camera_param_path'])

            # 2. world → fisheye 图像坐标
            keypoints_2d = camera.world2camera(keypoints_3d)

            if keypoints_2d.ndim == 1:
                keypoints_2d = keypoints_2d.reshape(-1, 2)
            depth_values = keypoints_3d[:, 2]

            # 3. 图像坐标 + 深度 → voxel 空间
            joints_voxel = fisheye2voxel(
                torch.from_numpy(keypoints_2d).float(),
                torch.from_numpy(depth_values).float(),
                voxel_size=voxel_size,
                heatmap_size=heatmap_size
            ).numpy()
            joints_voxel = np.clip(joints_voxel, 0, np.array(heatmap_size) - 1)

            # 4. 生成 limb heatmap
            limb_gt = generate_limb_heatmap_3d(
                torch.from_numpy(joints_voxel).float(),
                heatmap_size=heatmap_size,
                sigma=1  # or any appropriate sigma
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
