#  Copyright Jian Wang @ MPI-INF (c) 2023.

# regress the 3d heatmap under the fisheye camera view and give 3d pose prediction
import logging
import sys

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_upsample_layer,
                      constant_init, normal_init)
from scipy.ndimage import gaussian_filter


from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.models.builder import HEADS, build_loss
from mmpose.models.utils.ops import resize
from mmpose.utils.fisheye_camera.FishEyeCalibrated import FishEyeCameraCalibrated
from .voxel_fusion_modules import JointHeatmapEncoder_before, LimbHeatmapEncoder_before, PropagationFusionUnit,JointVoxelDecoder #代码在model/voxel_fusion_modules.py

LIMB_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14), (7, 11)
]

def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))    #每个关键点的热力图扁平化成 [B, J, D×H×W]
    heatmap3d = F.softmax(heatmap3d, 2)                                        #对每个关键点在整个体素空间上的分布做 softmax，变成概率分布
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))      #恢复为 [B, J, D, H, W]

    accu_x = heatmap3d.sum(dim=(2, 3))                                        #对每个关键点在 x 轴上的概率分布求和，计算边缘概率分布 # [B, J, W]
    accu_y = heatmap3d.sum(dim=(2, 4))                                        #对每个关键点在 y 轴上的概率分布求和，计算边缘概率分布 # [B, J, H]
    accu_z = heatmap3d.sum(dim=(3, 4))                                        #对每个关键点在 z 轴上的概率分布求和，计算边缘概率分布 # [B, J, D]
    
    device = heatmap3d.device
    #accu_x = accu_x * torch.arange(width, device=device).float().cuda()[None, None, :]
    #accu_y = accu_y * torch.arange(height, device=device).float().cuda()[None, None, :]
    #accu_z = accu_z * torch.arange(depth, device=device).float().cuda()[None, None, :]

    accu_x = accu_x * torch.arange(width, device=device).float()[None, None, :]
    accu_y = accu_y * torch.arange(height, device=device).float()[None, None, :]
    accu_z = accu_z * torch.arange(depth, device=device).float()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


@HEADS.register_module()
class FeatureMixLimbHeatmap3DNet(nn.Module):
    def __init__(self,
                 in_channels=768,
                 num_deconv_layers=2,
                 num_deconv_filters=(1024, 15 * 64),
                 num_deconv_kernels=(4, 4),
                 out_channels=15 * 64,
                 heatmap_shape=(64, 64, 64),
                 voxel_size=(2, 2, 2),
                 fisheye_model_path=None,
                 joint_num=15,
                 loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),
                 input_transform=None,
                 in_index=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs
                 ):
        super(FeatureMixLimbHeatmap3DNet, self).__init__()

    
        self.in_channels = in_channels
        self.deconv = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels)    
        #num_deconv_filters=[1024, 15 * 64]说的是有两层，第一层是1024，第二层是15 * 64，15是关节数，64是每个关节的通道数。
        #num_deconv_kernels=[4, 4]说的是两层都是4x4的卷积核。
        self.final_conv = nn.Conv2d(num_deconv_filters[-1], out_channels, kernel_size=1, stride=1, padding=0)
        #num_deconv_filters[-1]是第二层的通道数，out_channels是关节数*每个关节的通道数，15*64=960
        self.heatmap_shape = heatmap_shape
        self.fisheye_model = FishEyeCameraCalibrated(fisheye_model_path)
        self.joint_num = joint_num
        self.voxel_size = voxel_size
        self.loss = build_loss(loss_keypoint)
        

        self.limb_loss_weight = 0.1  # 🔧 可以调节，例如 0.1 ~ 1.0 之间  0.5

        #方法一第一阶段
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)    #自动学习权重比例

        #方法一第三阶段加入编码器
        #self.joint_encoder = JointHeatmapEncoder_before(in_channels=1, feature_dim=128)
        #self.limb_encoder = LimbHeatmapEncoder_before(in_channels=1, feature_dim=128)
        
        #方法一第二阶段加入cnn
        #self.fusion_conv = nn.Conv3d(30, 15, kernel_size=1)      #权重融合方法一的变形
        #方法二，通道注意力（channel-wise）
        #self.channel_attn_fc = nn.Sequential(
        #   nn.Linear(30, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 30),
        #    nn.Sigmoid()
        #)
        #方法三
        #self.joint_encoder = JointHeatmapEncoder_before(in_channels=1, feature_dim=128)
        #self.limb_encoder = LimbHeatmapEncoder_before(in_channels=1, feature_dim=128)
        #self.fusion_unit = PropagationFusionUnit(feature_dim=128)
        #self.voxel_decoder = JointVoxelDecoder(feature_dim=128, output_shape=(64, 64, 64))

        self.parent_map = {
            1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
            7: 1, 8: 7, 9: 8, 10: 9,
            11: 4, 12: 11, 13: 12, 14: 13
        }
        self.LIMB_CONNECTIONS = LIMB_CONNECTIONS
        self.limb_map = {tuple(sorted((a, b))): i for i, (a, b) in enumerate(self.LIMB_CONNECTIONS)}

        self.debug_iter_counter = 0                   #测试debug
        # === FeatureMixLimbHeatmap3DNet.__init__ ===gpt5建议
        self.alpha_param = nn.Parameter(torch.tensor(-2.0))  # 初始 alpha ~ 0.06
        # ====================================================







        self.input_transform = input_transform
        self.in_index = in_index
        logger = logging.getLogger()
        logger.info(f'Input transform: {self.input_transform}')
        logger.info(f'Input index: {self.in_index}')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        #新加的用来计算limb热力图的
        #self.limb_head = nn.Conv3d(joint_num, len(LIMB_CONNECTIONS), kernel_size=1)
        #limb热力图模型方案二：
        self.limb_head = nn.Sequential(
            nn.Conv3d(joint_num, 64, kernel_size=3, padding=1),  # 空间结构建模
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, len(LIMB_CONNECTIONS), kernel_size=1)  # 输出 limb heatmap
                )


    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list) or self.input_transform is None:
            return inputs                                                   #input_transform=None 或 inputs 不是列表,直接返回输入，不会进行多层融合或其他操作

        if self.input_transform == 'resize_concat':                         #表示从多个特征层中抽取，resize 为同一尺寸后拼接,常用于 FPN（特征金字塔）结构中。
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
            # suppose input size is 16 * 16
            B, C, H, W = inputs.shape
            if H != 16 or W != 16:
                # print('The input size is not 16 * 16, resize it to 16 * 16')
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear', align_corners=False)
        elif self.input_transform == 'multiple_select':                   #从多个层中抽取，分别使用，不拼接
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]                                #用于只选择某一层输出，常用于简单模型或 ViT，只用最后一层

        return inputs
    
    def _get_parent_feature(self, joint_feat, j):
        """
        返回第 j 个关节的 parent 关节特征。如果 j 是 root（例如 j=0），返回自身特征。
        """
        
        if j in self.parent_map:
            parent_j = self.parent_map[j]
        else:
            # 对于 root 点（例如 joint 0），没有 parent，返回自身特征作为占位
            parent_j = j
            
        return joint_feat[:, parent_j], parent_j
        
    def find_limb_index(self, parent, child):
        for i, (a, b) in enumerate(self.LIMB_CONNECTIONS):  # 你可以把它挂成类成员
            if (a == parent and b == child) or (a == child and b == parent):
                return i
        return None  # 没找到，返回 None


    def forward(self, img_feat):
        img_feat = self._transform_inputs(img_feat)                        #没有进行任何操作，直接返回输入
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        #self.deconv(img_feat) 是调用 self.deconv 模块，将特征图上采样成 3D 热力图，输出是 [B, 15 * 64, 64, 64]

        #view(-1, self.joint_num, self.heatmap_shape[0], self.heatmap_shape[1], self.heatmap_shape[2]) 是将输出展平为 [B, 15, 64, 64, 64]
        #joint_num=15, heatmap_shape=(64, 64, 64)

        #新加入的用来计算热力图的。这个热力图是在已有 joint 表达的结构空间中，学习 limb 表达。
        #limb_heatmap = self.limb_head(joint_3d_heatmap)
        limb_heatmap = self.limb_head(joint_3d_heatmap.detach())        #改过

        
        #############################测试
        #print("✅ limb_heatmap max:", limb_heatmap.max().item())
        #print("✅ limb_heatmap nonzero:", (limb_heatmap > 0).sum().item())

        #############################尝试加入特征融合
        #方法一，机构融合（由于形状一致不加卷积）自动学习融合权重
        if limb_heatmap.shape != joint_3d_heatmap.shape:
            limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)
        
        fused_heatmap = (1 - self.alpha) * joint_3d_heatmap + self.alpha * limb_heatmap
        joint_coord = soft_argmax_3d(fused_heatmap)

        #方法一，加编码器
        #if limb_heatmap.shape != joint_3d_heatmap.shape:
        #    limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)

        #joint_feat = self.joint_encoder(joint_3d_heatmap)  # [B, 15, 128]
        #limb_feat = self.limb_encoder(limb_heatmap) 
        
        #fused_heatmap = (1 - self.alpha) * joint_3d_heatmap + self.alpha * limb_heatmap
        #joint_coord = soft_argmax_3d(fused_heatmap)

        #方法一改进加入cnn
        # Resize limb_heatmap if necessary
        #if limb_heatmap.shape != joint_3d_heatmap.shape:
        #    limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)

        # Concatenate along channel dim: [B, 15 + 15, D, H, W] = [B, 30, D, H, W]
        #fused_input = torch.cat([joint_3d_heatmap, limb_heatmap], dim=1)

        # Apply 1x1x1 Conv to fuse features
        #fused_heatmap = self.fusion_conv(fused_input)  # output: [B, 15, D, H, W]

        # Predict joint coordinates from fused heatmap
        #joint_coord = soft_argmax_3d(fused_heatmap)

        #方法二，加入注意力
        # 保证 limb heatmap 尺寸一致
        #if limb_heatmap.shape != joint_3d_heatmap.shape:
        #    limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)

        # 拼接 joint 和 limb → [B, 30, D, H, W]
        #fused_input = torch.cat([joint_3d_heatmap, limb_heatmap], dim=1)  # 15 + 15 = 30

        # 通道注意力
        #gap = F.adaptive_avg_pool3d(fused_input, 1).view(fused_input.size(0), -1)  # [B, 30]
        #attn_weights = self.channel_attn_fc(gap).view(fused_input.size(0), 30, 1, 1, 1)  # [B, 30, 1, 1, 1]

        # 注意力加权
        #attn_applied = fused_input * attn_weights  # [B, 30, D, H, W]

        # 1x1x1卷积融合为 15通道 joint heatmap
        #fused_heatmap = self.fusion_conv(attn_applied)  # [B, 15, D, H, W]

        # soft-argmax 得到坐标
        #joint_coord = soft_argmax_3d(fused_heatmap)

        #方法三
        #joint_feat = self.joint_encoder(joint_3d_heatmap)  # [B, 15, 128]
        #limb_feat = self.limb_encoder(limb_heatmap)        # [B, 15, 128]

        ##print('joint_feat=====',joint_feat.shape)
        ##print('limb_feat=====',limb_feat.shape)


        # joint_feat: [B, J, C]
        # limb_feat:  [B, L, C]
        
        #改成批量化改法(新的2025.8.5)
        #B, J, C = joint_feat.shape

        # ====== 1. 批量 parent_feat 获取 ======
        #parent_indices = torch.tensor(
        #    [self.parent_map.get(j, j) for j in range(J)],  # 没有parent就取自己
        #    device=joint_feat.device
        #)  # [J]
        #parent_feat = joint_feat.gather(
        #    1, parent_indices.view(1, J, 1).expand(B, J, C)
        #)  # [B, J, C]

        # ====== 2. 批量 limb_feat 获取 ======
        #limb_indices = []
        #for j in range(J):
        #    parent_j = self.parent_map.get(j, j)
        #    limb_key = tuple(sorted((parent_j, j)))
        #    if limb_key in self.limb_map:
        #        limb_indices.append(self.limb_map[limb_key])
        #    else:
        #        limb_indices.append(-1)  # -1 表示没有 limb

        #limb_indices = torch.tensor(limb_indices, device=joint_feat.device)  # [J]
        #=====================================测试debug=====================================
        #if self.debug_iter_counter < 1:  # 只打印一次
        #    for j in range(J):
        #        parent_j = self.parent_map.get(j, j)
        #        # 查找 limb index
        #        limb_idx = self.limb_map.get(tuple(sorted((parent_j, j))), None)
        #        print(f"[DEBUG] joint {j:2d} -> parent {parent_j:2d} -> limb_idx {limb_idx}")
        #        if limb_idx is not None:
        #            a, b = LIMB_CONNECTIONS[limb_idx]
        #            if sorted((a, b)) != sorted((parent_j, j)):
        #                print(f"❌ MISMATCH: limb {limb_idx} = {a,b}, expected ({parent_j},{j})")
        #            else:
        #                print(f"✅ MATCH: limb {limb_idx} = {a,b}")
        #    self.debug_iter_counter += 1
        #==========================================================================



        # 对有 limb 的位置 gather，没有 limb 的用 0 填充
        #valid_mask = limb_indices != -1
        #limb_indices_clamped = limb_indices.clone()
        #limb_indices_clamped[~valid_mask] = 0  # 防止 gather 出界
        #limb_feat_batch = limb_feat.gather(
        #    1, limb_indices_clamped.view(1, J, 1).expand(B, J, C)
        #)  # [B, J, C]
        #limb_feat_batch[~valid_mask] = 0  # 没有 limb 的置 0
        #limb_feat_batch[~valid_mask.view(1, J, 1).expand(B, J, C)] = 0

        #print("Step Debug:")
        #print("  limb_feat_batch mean/std:", limb_feat_batch.mean().item(), limb_feat_batch.std().item())

        # ====== 3. 一次性融合 ======（取消其他非测试注释可还原）
        #fused_feat = self.fusion_unit(parent_feat, joint_feat, limb_feat_batch)  # [B, J, C]

        #=====================================测试debug=====================================
        #self.debug_iter_counter += 1
        
        #if self.debug_iter_counter > 1000:  # 超过1000 iter后才打印
        #    fused_feat, debug_info = self.fusion_unit(
        #        parent_feat, joint_feat, limb_feat_batch, return_debug=True
        #    )
        #    if self.debug_iter_counter % 500 == 0:  # 每500 iter打印一次
        #        print(f"[Iter {self.debug_iter_counter}] Step Debug:")
        #        print(f"  limb_feat_batch mean/std: {limb_feat_batch.mean().item()} {limb_feat_batch.std().item()}")
        #        print(debug_info)
        #        #print(f"  joint_feat std: {joint_feat.std().item()} fused_feat std: {fused_feat.std().item()}")
        #else:
        #    fused_feat = self.fusion_unit(parent_feat, joint_feat, limb_feat_batch)
        #==========================================================================

        #fused_feat, debug_info = self.fusion_unit(parent_feat, joint_feat, limb_feat_batch, return_debug=True)
        #print(debug_info)
        
        #print("  joint_feat std:", joint_feat.std().item(), 
        #"fused_feat std:", fused_feat.std().item())
        
        #fused_heatmap = self.voxel_decoder(fused_feat)        # [B, J, 64, 64, 64]
        #joint_coord = soft_argmax_3d(fused_heatmap)           # [B, J, 3]

        # === forward 末尾（替换你现在两行）===
        #residual = self.voxel_decoder(fused_feat)
        #alpha = torch.sigmoid(self.alpha_param) * 0.1
        #fused_heatmap = joint_3d_heatmap + alpha * residual
        #fused_heatmap = joint_3d_heatmap.detach() + alpha * residual
        ##joint_coord = soft_argmax_3d(residual)
        #joint_coord = soft_argmax_3d(fused_heatmap)

        # === 监控用统计 ===训练要打开##和下面的
        #if self.debug_iter_counter % 500 == 0:  # 你已有的频率判断
        #    with torch.no_grad():
        #        base_std = joint_3d_heatmap.std().item()
        #        res_mean = residual.mean().item()
        #        res_std  = residual.std().item()
        #        fuse_std = fused_heatmap.std().item()
        #        delta_l2 = (residual ** 2).mean().sqrt().item()
        #        print(f"[Iter {self.debug_iter_counter}] Residual Debug:")
        #        print(f"α={alpha.item():.4f} | base_std={base_std:.4f} | res_mean/std={res_mean:.4f}/{res_std:.4f} | fused_std={fuse_std:.4f} | delta_l2={delta_l2:.4f}")
        #==========================================================================


        

        '''
        #原始可用的
        # 引导融合（基于 PropagationFusionUnit）
        fused_feat = []
        for j in range(self.joint_num):
            parent_feat, parent_j = self._get_parent_feature(joint_feat, j)
            child_feat = joint_feat[:, j]

            limb_key = tuple(sorted((parent_j, j)))
            #print('limb_map',self.limb_map)
            limb_idx = self.limb_map.get(limb_key)
            if limb_idx is not None:
                limb_feat_j = limb_feat[:, limb_idx]
            else:
                #print('limb_idx is  None---没有找到对应limb')
                limb_feat_j = torch.zeros_like(child_feat)

            
            fused = self.fusion_unit(parent_feat, child_feat, limb_feat_j)
            fused_feat.append(fused.unsqueeze(1))
        fused_feat = torch.cat(fused_feat, dim=1)  # [B, J, C]
        ##print('fused_feat=',fused_feat.shape)

        # voxel解码器得到 [B, J, 64, 64, 64]
        fused_heatmap = self.voxel_decoder(fused_feat)
        ##print('fused_heatmap=',fused_heatmap.shape)
        joint_coord = soft_argmax_3d(fused_heatmap)
        '''

        
        #原始代码（不用特征融合的时候记得取消注释）
        #joint_coord = soft_argmax_3d(joint_3d_heatmap)

        # the joint coord is under 64 * 64 * 64 space, we need to convert it to real world space
        resize_z = self.voxel_size[2] / self.heatmap_shape[2]
        # resize the depth to real world space
        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z
        # resize x and y to image space
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024

        # convert joint coord in fisheye space to joint coord in camera space
        joint_coord = self.fisheye2camera(joint_coord)       #图像上的坐标转换为真实 3D 空间的坐标

        #return joint_coord
        return {
                'preds': joint_coord,
                'limb_heatmap': limb_heatmap
               }

    def fisheye2camera(self, joint_coord):
        # joint_coord: [batch_size, joint_num, 3]
        # joint_coord_cam: [batch_size, joint_num, 3]
        batch_size = joint_coord.shape[0]
        joint_coord_xy = joint_coord[:, :, :2].view(batch_size * self.joint_num, 2)
        joint_coord_z = joint_coord[:, :, 2].view(batch_size * self.joint_num)
        joint_coord_cam = self.fisheye_model.camera2world_pytorch(joint_coord_xy, joint_coord_z)
        joint_coord_cam = joint_coord_cam.view(batch_size, self.joint_num, 3)
        joint_coord_cam = joint_coord_cam.contiguous()
        return joint_coord_cam

    def get_loss(self, output, keypoints_3d, keypoint_3d_visible=None, **kwargs):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_keypoint_pos: 3

        Args:
            output (torch.Tensor[N,K,3): Output keypoints.
            keypoints_3d (torch.Tensor[N,K,3]): Target keypoints.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert keypoints_3d.dim() == 3
        #losses['mpjpe_loss'] = self.loss(output, keypoints_3d, keypoint_3d_visible[:, :, None]) #原始的loss
        losses['mpjpe_loss'] = self.loss(output['preds'], keypoints_3d, keypoint_3d_visible[:, :, None])

        #增加loss
        pred, gt = output['preds'], keypoints_3d                   # [B, J, 3]
        angle_loss, len_loss = 0., 0.
        for (a, b) in LIMB_CONNECTIONS:
            v_pred = pred[:, b] - pred[:, a]
            v_gt   = gt[:, b]   - gt[:, a]
            angle_loss += (1 - F.cosine_similarity(v_pred, v_gt, dim=-1)).mean()
            len_loss   += F.l1_loss(v_pred.norm(dim=-1), v_gt.norm(dim=-1))
        losses['bone_dir'] = 1e-3 * angle_loss
        losses['bone_len'] = 1e-3 * len_loss



        #新加入的用于计算limbloss的
        if 'limb_heatmap' in output and 'target_limb_heatmap' in kwargs:
            pred_limb = output['limb_heatmap']
            target_limb = kwargs['target_limb_heatmap']

            #############新！用于全部limb热力图归一化，修改这里的时候记得修改数据集加载部分的归一化
            #B, C, D, H, W = pred_limb.shape
            #pred_limb = pred_limb.view(B, C, -1)
            #max_vals = pred_limb.max(dim=2, keepdim=True)[0].clamp(min=1e-6)  # 防止除以0
            #pred_limb = pred_limb / max_vals
            #pred_limb = pred_limb.view(B, C, D, H, W)
            #############

            limb_loss = F.mse_loss(pred_limb, target_limb)

            # ======== 验证通道顺序是否一致 ========（测试bug）
            if not hasattr(self, "_debug_printed"):
                with torch.no_grad():
                    B, C, D, H, W = pred_limb.shape
                    corr = torch.zeros(C, C, device=pred_limb.device)
                    for i in range(C):
                        for j in range(C):
                            corr[i, j] = F.cosine_similarity(
                                pred_limb[:, i].reshape(B, -1),
                                target_limb[:, j].reshape(B, -1),
                                dim=1
                            ).mean()
                    print("🔍 limb_heatmap 预测与 GT 的通道相关性矩阵：")
                    print(corr.cpu().numpy())
                self._debug_printed = True
            # ====================================


            losses['limb_loss'] = self.limb_loss_weight * limb_loss
        
        #########################测试
        #if 'limb_loss' in losses:
        #    print("✅ limb_loss:", losses['limb_loss'].item())


        return losses

    def get_accuracy(self, output, keypoints_3d, keypoint_3d_visible=None):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        """

        accuracy = dict()
        pred = output['preds']
        N, K, _ = pred.shape

        mpjpe = keypoint_mpjpe(
            pred.detach().cpu().numpy(),
            keypoints_3d.detach().cpu().numpy(),
            mask=keypoint_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        return accuracy

    def inference_model(self, x, flip_pairs=None,**kwargs):  #增加了,**kwargs
        #"""
        #原始的代码
        #Inference function.

        #Returns:
        #    output_heatmap (np.ndarray): Output heatmaps.

        #Args:
        #    x (torch.Tensor[N,K,H,W]): Input features.
        #    flip_pairs (None | list[tuple]):
        #        Pairs of keypoints which are mirrored.
        
        img_feat = self._transform_inputs(x)
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        limb_heatmap = self.limb_head(joint_3d_heatmap) 
        
        #方法一
        if limb_heatmap.shape != joint_3d_heatmap.shape:
            limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)
        
        #print('a=',self.alpha)
        
        fused_heatmap = (1 - self.alpha) * joint_3d_heatmap + self.alpha * limb_heatmap
        joint_coord = soft_argmax_3d(fused_heatmap)

        #方法一改进加入cnn
        # Resize limb_heatmap if necessary
        #if limb_heatmap.shape != joint_3d_heatmap.shape:
        #    limb_heatmap = F.interpolate(limb_heatmap, size=joint_3d_heatmap.shape[2:], mode='trilinear', align_corners=False)

        # Concatenate along channel dim: [B, 15 + 15, D, H, W] = [B, 30, D, H, W]
        #fused_input = torch.cat([joint_3d_heatmap, limb_heatmap], dim=1)

        # Apply 1x1x1 Conv to fuse features
        #fused_heatmap = self.fusion_conv(fused_input)  # output: [B, 15, D, H, W]

        # Predict joint coordinates from fused heatmap
        #joint_coord = soft_argmax_3d(fused_heatmap)



        #方法三
        # == 与 forward 同步：先算 limb，再两路编码 ============================================gpt5，融合模块推添加
        ##limb_heatmap = self.limb_head(joint_3d_heatmap)
        #limb_heatmap = self.limb_head(joint_3d_heatmap)
        #joint_feat = self.joint_encoder(joint_3d_heatmap)   # [B, J, C]
        #limb_feat  = self.limb_encoder(limb_heatmap)        # [B, L, C]（L = len(LIMB_CONNECTIONS)）

        #B, J, C = joint_feat.shape

        # == 批量 parent 特征 ==
        #parent_indices = torch.tensor(
        #    [self.parent_map.get(j, j) for j in range(J)],
        #    device=joint_feat.device, dtype=torch.long
        #)  # [J]
        #parent_feat = joint_feat.gather(1, parent_indices.view(1, J, 1).expand(B, J, C))  # [B,J,C]

        # == 批量 limb 特征 ==
        # 用 limb_map 把 (parent, child) → limb 索引；没找到的给 -1
        #limb_indices = []
        #for j in range(J):
        #    p = self.parent_map.get(j, j)
        #    limb_indices.append(self.limb_map.get(tuple(sorted((p, j))), -1))
        #limb_indices = torch.tensor(limb_indices, device=limb_feat.device, dtype=torch.long)  # [J]
        #valid = limb_indices >= 0
        #idx_clamped = limb_indices.clamp(min=0)
        #limb_feat_batch = limb_feat.gather(1, idx_clamped.view(1, J, 1).expand(B, J, C))     # [B,J,C]
        #limb_feat_batch = limb_feat_batch.masked_fill(~valid.view(1, J, 1).expand(B, J, C), 0)

        # == PFU 融合 + 体素解码 + 残差叠加（与 forward 保持一致）==
        #fused_feat = self.fusion_unit(parent_feat, joint_feat, limb_feat_batch)
        #residual = self.voxel_decoder(fused_feat)
        #alpha = torch.sigmoid(self.alpha_param) * 0.1
        #fused_heatmap = joint_3d_heatmap + alpha * residual   # 推理不需要 detach

        # == 用融合后的热图回归坐标 ==
        #joint_coord = soft_argmax_3d(residual)

        #joint_coord = soft_argmax_3d(fused_heatmap)
        # ===================================================



        #原始方法只有这一行！！！
        #joint_coord = soft_argmax_3d(joint_3d_heatmap)

        joint_coord_in_voxel = joint_coord.detach().clone()

        # the joint coord is under 64 * 64 * 64 space, we need to convert it to real world space
        resize_z = self.voxel_size[2] / self.heatmap_shape[2]
        # resize the depth to real world space
        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z
        # resize x and y to image space
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024

        # convert joint coord in fisheye space to joint coord in camera space
        joint_coord = self.fisheye2camera(joint_coord)
        result = {'keypoints_pred': joint_coord.detach().cpu().numpy()}
        if 'return_heatmap' in self.test_cfg and self.test_cfg['return_heatmap'] is True:
            result['heatmap'] = joint_3d_heatmap.detach().cpu().numpy()

        if 'return_confidence' in self.test_cfg and self.test_cfg['return_confidence'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_voxel = joint_coord_in_voxel.detach()
            batch_size, joint_num, depth, height, width = joint_3d_heatmap_confidence.shape
            # use grid_sample to get the confidence
            # joint_voxel shape: (batch_size, joint_number, 3)
            # add gaussian filter
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            joint_3d_heatmap_confidence = torch.from_numpy(joint_3d_heatmap_confidence).to(joint_voxel.device)

            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size * joint_num, 1, depth, height,
                                                                           width)

            joint_voxel = joint_voxel.view(batch_size * joint_num, 1, 1, 1, 3)
            assert self.heatmap_shape[0] == self.heatmap_shape[1] == self.heatmap_shape[2]
            joint_voxel = joint_voxel / self.heatmap_shape[2] * 2 - 1
            joint_3d_heatmap_confidence = torch.nn.functional.grid_sample(
                joint_3d_heatmap_confidence, joint_voxel, align_corners=False)
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size, joint_num)
            result['keypoint_confidence'] = joint_3d_heatmap_confidence.detach().cpu().numpy()
            # print(result['keypoint_confidence'])
        if 'return_2d_heatmap' in self.test_cfg and self.test_cfg['return_2d_heatmap'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            # convert 3d heatmap to 2d heatmap alone the z axis
            joint_2d_heatmap_confidence = joint_3d_heatmap_confidence.sum(axis=2)
            result['heatmap_2d'] = joint_2d_heatmap_confidence

        #if self.test_cfg.get('save_limb_heatmap', False):
        #    save_dir = self.test_cfg.get('limb_heatmap_save_dir', './outputs/limb_heatmaps')
        #    os.makedirs(save_dir, exist_ok=True)

        #    with torch.no_grad():
        #        limb_heatmap = self.limb_head(joint_3d_heatmap)  # shape: [B, L, D, H, W]

        #    limb_pred = limb_heatmap.detach().cpu().numpy()
        #    np.save(os.path.join(save_dir, 'pred_limb_heatmap_sample0.npy'), limb_pred[0])

                # ✅ 判断 target_limb_heatmap 是否存在且不为 None
        #    if kwargs is not None and 'target_limb_heatmap' in kwargs and kwargs['target_limb_heatmap'] is not None:
        #        gt_limb = kwargs['target_limb_heatmap'].detach().cpu().numpy()
        #        np.save(os.path.join(save_dir, 'gt_limb_heatmap_sample0.npy'), gt_limb[0])
        #        print(f"✅ Saved predicted & GT limb heatmap to {save_dir}")
        #    else:
        #        print("⚠️ Warning: 'target_limb_heatmap' is not available in kwargs or is None.")
        #    #sys.exit()
                
                
        """

        #新改的用于保存limb热力图的代码
        img_feat = self._transform_inputs(x)
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        joint_coord = soft_argmax_3d(joint_3d_heatmap)

        joint_coord_in_voxel = joint_coord.detach().clone()

        # the joint coord is under 64 * 64 * 64 space, we need to convert it to real world space
        resize_z = self.voxel_size[2] / self.heatmap_shape[2]
        # resize the depth to real world space
        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z
        # resize x and y to image space
        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024

        # convert joint coord in fisheye space to joint coord in camera space
        joint_coord = self.fisheye2camera(joint_coord)
        result = {'keypoints_pred': joint_coord.detach().cpu().numpy()}
        if 'return_heatmap' in self.test_cfg and self.test_cfg['return_heatmap'] is True:
            result['heatmap'] = joint_3d_heatmap.detach().cpu().numpy()

        if 'return_confidence' in self.test_cfg and self.test_cfg['return_confidence'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_voxel = joint_coord_in_voxel.detach()
            batch_size, joint_num, depth, height, width = joint_3d_heatmap_confidence.shape
            # use grid_sample to get the confidence
            # joint_voxel shape: (batch_size, joint_number, 3)
            # add gaussian filter
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            joint_3d_heatmap_confidence = torch.from_numpy(joint_3d_heatmap_confidence).to(joint_voxel.device)

            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size * joint_num, 1, depth, height,
                                                                           width)

            joint_voxel = joint_voxel.view(batch_size * joint_num, 1, 1, 1, 3)
            assert self.heatmap_shape[0] == self.heatmap_shape[1] == self.heatmap_shape[2]
            joint_voxel = joint_voxel / self.heatmap_shape[2] * 2 - 1
            joint_3d_heatmap_confidence = torch.nn.functional.grid_sample(
                joint_3d_heatmap_confidence, joint_voxel, align_corners=False)
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.view(batch_size, joint_num)
            result['keypoint_confidence'] = joint_3d_heatmap_confidence.detach().cpu().numpy()
            # print(result['keypoint_confidence'])
        if 'return_2d_heatmap' in self.test_cfg and self.test_cfg['return_2d_heatmap'] is True:
            # calculate confidence for each joint
            # joint_voxel shape: (batch_size, joint_number, 3)
            # heatmap shape: (batch_size, joint_number, Depth, Height, Width)
            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))
            # convert 3d heatmap to 2d heatmap alone the z axis
            joint_2d_heatmap_confidence = joint_3d_heatmap_confidence.sum(axis=2)
            result['heatmap_2d'] = joint_2d_heatmap_confidence
            # ✅ 追加 limb heatmap 保存逻辑
             #       ✅ 追加 limb heatmap 保存逻辑
        if self.test_cfg.get('save_limb_heatmap', False):
            save_dir = self.test_cfg.get('limb_heatmap_save_dir', './outputs/limb_heatmaps')
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                limb_heatmap = self.limb_head(joint_3d_heatmap)  # shape: [B, L, D, H, W]

            limb_pred = limb_heatmap.detach().cpu().numpy()
            np.save(os.path.join(save_dir, 'pred_limb_heatmap_sample0.npy'), limb_pred[0])

            # ✅ 判断 target_limb_heatmap 是否存在且不为 None
            if kwargs is not None and 'target_limb_heatmap' in kwargs and kwargs['target_limb_heatmap'] is not None:
                gt_limb = kwargs['target_limb_heatmap'].detach().cpu().numpy()
                np.save(os.path.join(save_dir, 'gt_limb_heatmap_sample0.npy'), gt_limb[0])
                print(f"✅ Saved predicted & GT limb heatmap to {save_dir}")
            else:
                print("⚠️ Warning: 'target_limb_heatmap' is not available in kwargs or is None.")
        """ 


        return result

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        normal_init(self.final_conv, std=0.001, bias=0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),     #通过改这个就能构建不同的上采样层
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
