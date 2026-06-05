# Copyright (c) OpenMMLab. All rights reserved.
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
    #    print("✅ 有写入高斯值")
    #else:
    #    print("❌ g 全是 0，没写入任何值")


    volume[zz, yy, xx] += g
    #print(f"中心坐标：{center.tolist()}, 体素范围: {volume.shape}")
#'''

def generate_limb_heatmap_3d(joints_3d, heatmap_size=(64, 64, 64), sigma=1.5):      #sigma=[2.0, 0.6, 0.6]
    num_limbs = len(LIMB_CONNECTIONS)
    D, H, W = heatmap_size
    heatmaps = torch.zeros((num_limbs, D, H, W), dtype=torch.float32)     #初始化热力图
    #print(f"draw_gaussian_3d() 执行次数统计")
    draw_count = 0


    for idx, (j1, j2) in enumerate(LIMB_CONNECTIONS):
        p1 = joints_3d[j1]
        p2 = joints_3d[j2]

        if torch.any(torch.isnan(p1)) or torch.any(torch.isnan(p2)):
            print(f"⚠️ limb {idx} skipped due to NaN")
            continue

        dist = torch.norm(p2 - p1).item()
        n_points = max(int(dist * 2), 1)
        #print(f"Limb {idx}: dist = {dist:.2f}, n_points = {n_points}")

        for t in torch.linspace(0, 1, steps=n_points):
            pt = p1 * (1 - t) + p2 * t
            #print(f"   ↪ draw at pt: {pt.tolist()}")
            #draw_anisotropic_gaussian_3d(heatmaps[idx], pt, sigma)
            draw_gaussian_3d(heatmaps[idx], pt, sigma)
            
            #draw_count += 1
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
class LimbRenderpeopleMixamoDataset(Dataset):        #用于 从 RenderPeople+Mixamo 合成数据集中加载和处理 3D 姿态估计训练样本的自定义类。
    allowed_metrics = ['pa-mpjpe', 'mpjpe', 'ba-mpjpe']

    def __init__(self,
                 ann_file,      
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dset='renderpeople_old',
                 test_mode=False):

        self.ann_file = ann_file                        #标注文件路径（.pkl）
        self.img_prefix = img_prefix                    #图像文件夹路径,最终图像路径为：img_prefix + image_file_name
        self.data_cfg = copy.deepcopy(data_cfg)         #数据配置
        self.pipeline = pipeline                        #数据处理管道
        self.test_mode = test_mode                      #是否为测试模式,为 True 时表示测试模式，不加载训练标签，只加载测试标签
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
            # old renderpeople dataset format
            data_list = data['data_list']
        elif self.dset == 'renderpeople':
            # new renderpeople data format
            data_list = []
            data_raw_list = data['data_list']
            for identity_name, identity_data in data_raw_list.items():
                print(f'load identity: {identity_name}')
                for seq_name, seq_data in tqdm(identity_data.items()):
                    data_list.extend(seq_data)
        else:
            raise Exception('dset type is incorrect')

        if self.test_mode is True:
            # select part of the dataset for evaluation
            data_list = data_list[:200]

        #######################测试
        #data_list = data_list[:30]
        
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
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """
        with open(res_file, 'rb') as fin:
            preds = pickle.load(fin)
        assert len(preds) == len(self.data_info)

        pred_joints_3d = [pred['keypoints'] for pred in preds]
        gt_joints_3d = [item['joints_3d'] for item in self.data_info]

        pred_joints_3d = np.array(pred_joints_3d)
        gt_joints_3d = np.array(gt_joints_3d)

        assert len(pred_joints_3d[0]) == len(gt_joints_3d[0])

        # we only evaluate on 14 lsp joints
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
        """Get data sample."""
        result = {}
        data = self.data_info[idx]
        '''
        data_info[idx]:
        {
            'image_file': 'renderpeople/img_00123.jpg',
            'keypoints_3d': np.array([[x1, y1, z1], ..., [x15, y15, z15]]),
            'keypoints_3d_visible': np.array([1, 1, ..., 1]),
            'camera_param': {...},
            ...
        }

        '''
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

            # if set only hand = True, then only use hand keypoints
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
            # convert from renderpeople to smplx joint
            keypoints3d[self.model_idxs] = keypoints[self.dst_idxs]
            keypoints3d_visible[self.model_idxs] = 1
            result['keypoints_3d'] = keypoints3d
            result['keypoints_3d_visible'] = keypoints3d_visible


        result['body_pose'] = np.zeros((21 * 3), dtype=np.float32)
        result['betas'] = np.zeros((10,), dtype=np.float32)

        result['has_smpl'] = 0

        # return bbox of hands
        # return voxel-level limb heatmap for supervision
        if 'keypoints_3d' in result:
            keypoints_3d = result['keypoints_3d']  
            #print('keypoints_3d====',keypoints_3d)
            keypoints_3d = np.asarray(keypoints_3d).astype(np.float32)
            #print('keypoints_3d2====',keypoints_3d)

            heatmap_size = tuple(self.ann_info.get('heatmap_size', (64, 64, 64)))
            if len(heatmap_size) == 2:
                heatmap_size = (heatmap_size[0], heatmap_size[1], heatmap_size[1])
            voxel_size = tuple(self.ann_info.get('voxel_size', (2, 2, 2)))

            # === Step 1: 加载相机模型 ===
            
            camera = FishEyeCameraCalibrated(self.ann_info['camera_param_path'])

            # === Step 2: world → camera image coordinates ===
            keypoints_2d = camera.world2camera(keypoints_3d)  # [J, 2]
            #print('keypoints_2d====',keypoints_2d)
            #print('keypoints_2d.shape====',keypoints_2d.shape)

            if keypoints_2d.ndim == 1:
                keypoints_2d = keypoints_2d.reshape(-1, 2)

            # === Step 3: 用 GT 的 z 值作为 depth（伪 depth map）===
            depth_values = keypoints_3d[:, 2]  # [J]
            #print('depth_values====',depth_values)

            # === Step 4: 通过 fisheye2voxel 转换到 voxel 空间 ===
            
            joints_voxel = fisheye2voxel(
                torch.from_numpy(keypoints_2d).float(),
                torch.from_numpy(depth_values).float(),
                voxel_size=voxel_size,
                heatmap_size=heatmap_size
            ).numpy()  # [J, 3]
            #print('joints_voxel====',joints_voxel)

            joints_voxel = np.clip(joints_voxel, 0, np.array(heatmap_size) - 1)
            #print('joints_voxel_after_clip====',joints_voxel)
            #print('joints_voxel_after_clip====')
            #print(f"[DEBUG] GT 3D joints = {keypoints_3d.shape[0]}, projected voxel joints = {joints_voxel.shape[0]}")


            # === Step 5: 生成 GT limb heatmap ===
            limb_gt = generate_limb_heatmap_3d(
                torch.from_numpy(joints_voxel).float(),
                heatmap_size=heatmap_size,
                sigma=1
                #sigma=[1.5, 0.3, 0.3]
            )
            #print('返回热力图结果=',limb_gt)
            torch.set_printoptions(precision=5, sci_mode=False)
            #print(limb_gt[14])
            #print(f"Limb 14: min = {limb_gt[14].min()}, mean = {limb_gt[14].mean()}, max = {limb_gt[14].max()}")


            result['target_limb_heatmap'] = limb_gt  # 添加到结果中



        return result

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_info)

    def __getitem__(self, idx):
        """Get a sample with given index."""
        results = copy.deepcopy(self.prepare_data(idx))
        results['ann_info'] = self.ann_info

        ###################################测试
        #if 'target_limb_heatmap' in results:
        #    print("✅ target_limb_heatmap max:", results['target_limb_heatmap'].max().item())
        #    print("✅ target_limb_heatmap nonzero:", (results['target_limb_heatmap'] > 0).sum().item())



        return self.pipeline(results)
        #return results
