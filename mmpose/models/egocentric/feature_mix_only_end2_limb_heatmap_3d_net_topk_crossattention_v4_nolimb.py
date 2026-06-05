#  Copyright Jian Wang @ MPI-INF (c) 2023.

# regress the 3d heatmap under the fisheye camera view and give 3d pose prediction
import logging
import sys
import time
import os
import math
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
from .voxel_fusion_modules import JointHeatmapEncoder, LimbHeatmapEncoder, DynamicLimbSelector,JointVoxelDecoder #代码在model/voxel_fusion_modules.py

LIMB_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (1, 7), (4, 11), (7, 8), (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14), (7, 11)
]

parent_map = {
    1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
    7: 1, 8: 7, 9: 8, 10: 9,
    11: 4, 12: 11, 13: 12, 14: 13
}
J = 15
L = 15
joint_to_limb_mask = torch.zeros(J, L, dtype=torch.bool)
for l, (a, b) in enumerate(LIMB_CONNECTIONS):
    joint_to_limb_mask[a, l] = True
    joint_to_limb_mask[b, l] = True

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


class CrossTGFI3D(nn.Module):
    """
    和第一版相比这个第二版本改变了残差后的k值的写回方式
    """
    def __init__(self, d_model=64, nhead=4, k_joint=32, k_limb=64,
                 bidirectional=True, restrict_to_adjacent=True,
                 diffusion_radius=1,zero_limb_for_cross=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.kj = k_joint
        self.kl = k_limb
        self.bidir = bidirectional
        self.restrict = restrict_to_adjacent
        self.diffusion_radius = diffusion_radius
        self.zero_limb_for_cross = zero_limb_for_cross

        # 标量(热度) + 坐标(3) -> token embedding
        self.embed = nn.Linear(1 + 3, d_model)
        self.proj_out = nn.Linear(d_model, 1)  # token -> 标量(写回的增量)
        self.value_head = nn.Linear(d_model, 1) #整个数值预测头
        self.gate_head  = nn.Linear(d_model, 1)

        self.attn_jq_lkv = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if bidirectional:
            self.attn_lq_jkv = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 基于距离的相对位置偏置（简单可学习尺度）
        self.dist_scale = nn.Parameter(torch.tensor(2.0))#0.5#2.0!!#10
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self._dbg_counter = 0
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long))
        self.delta_warm_iters = 1000#3000 #1000!!

    @staticmethod
    def _topk_tokenize_3d(x, k):
        """
        x: [B, C, D, H, W] -> (vals:[B,C,K,1], idx:[B,C,K], pos:[B,C,K,3] 归一化到[0,1])
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        flat = x.view(B, C, N)
        val, idx = torch.topk(flat, min(k, N), dim=-1)  # [B,C,K]                #取topk
        z = torch.div(idx, H * W, rounding_mode='floor').float()
        y = torch.div(idx.remainder(H * W), W, rounding_mode='floor').float()
        x_ = idx.remainder(W).float()
        # 归一化坐标，作为 TGFI 的 Lp 保真映射依据
        pos = torch.stack([x_ / (W - 1 + 1e-6),
                           y / (H - 1 + 1e-6),
                           z / (D - 1 + 1e-6)], dim=-1)  # [B,C,K,3]
        return val.unsqueeze(-1), idx, pos  # val shape [B,C,K,1]


    @staticmethod
    def _scatter_writeback(delta_tokens, idx, shape, radius=1):
        """
        delta_tokens: [B,C,K,1], idx: [B,C,K], 写回到 [B,C,D,H,W]
        radius > 0 时做局部 3D 平均扩散，缓解仅点写回的 alias。
        """
        B, C, K, _ = delta_tokens.shape
        D, H, W = shape
        N = D * H * W
        out = torch.zeros(B, C, N, device=delta_tokens.device, dtype=delta_tokens.dtype)
        out.scatter_add_(dim=2, index=idx, src=delta_tokens.squeeze(-1))
        out = out.view(B, C, D, H, W)
        if radius and radius > 0:
            r = int(radius)
            out = F.avg_pool3d(out, kernel_size=2*r+1, stride=1, padding=r)
        return out

    def _pairwise_dist(self, P, Q):
        """
        P: [B, NK, 3], Q: [B, ML, 3] -> dist^2: [B, NK, ML] 关节 token 的归一化坐标，limb token 的归一化坐标
        总的说是计算了关节 token 和 limb token 的 所有两两欧式距离平方
        """
        # (x - y)^2 = x^2 + y^2 - 2xy
        P2 = (P * P).sum(-1, keepdim=True)          # [B,NK,1]
        Q2 = (Q * Q).sum(-1, keepdim=True).transpose(1, 2)  # [B,1,ML]
        dist2 = P2 + Q2 - 2.0 * P @ Q.transpose(1, 2)
        return dist2.clamp_min_(0.0)

    def forward(self, joint_hm, limb_hm, joint_to_limb_mask=None):
        """
        joint_hm: [B, J, D,H,W]
        limb_hm:  [B, L, D,H,W]
        joint_to_limb_mask: [J, L] (bool) 可选，仅允许相邻肢体参与注意力
        """
        #print('joint_hm=',joint_hm)
        #print('limb_hm=',limb_hm)
        B, J, D, H, W = joint_hm.shape
        L = limb_hm.shape[1]

        vj, ij, pj = self._topk_tokenize_3d(joint_hm, self.kj)  # [B,J,K,1], [B,J,K], [B,J,K,3]
        vl, il, pl = self._topk_tokenize_3d(limb_hm, self.kl)

        # tokens 展平到 [B, (J*K), d] / [B, (L*K), d]（将top-k值与坐标concat起来）
        tj = self.embed(torch.cat([vj, pj], dim=-1)).view(B, J*self.kj, self.d_model)
        tl = self.embed(torch.cat([vl, pl], dim=-1)).view(B, L*self.kl, self.d_model)
        
        if self.zero_limb_for_cross:
            tl = tl.detach() * 0.0   # ★ 这里直接把 K/V 干掉
        else:
            print('zero_limb_for_cross is not 0000')
            tl = tl

        # 位置偏置（负距离作为 logit 偏置）
        Pj = pj.view(B, J*self.kj, 3)
        Pl = pl.view(B, L*self.kl, 3)
        dist2 = self._pairwise_dist(Pj, Pl)  # [B, J*K, L*K]
        # ——[NEW]：让 dist_scale 始终非负，数值更稳——
        dist_scale = F.softplus(self.dist_scale)  # >=0
        pos_bias   = -dist2 * dist_scale          # [B, J*K, L*K]

        if self.restrict and (joint_to_limb_mask is not None):
            # m: 允许注意的二值掩码 [B, J*K, L*K]
            m = joint_to_limb_mask[None, :, None, :, None].to(torch.bool).to(tj.device)  # [1,J,1,L,1]
            m = m.expand(B, J, self.kj, L, self.kl).reshape(B, J*self.kj, L*self.kl)

            # ——[NEW]：保证每个 query 至少有一个 key——
            row_all_masked = ~m.any(dim=-1)                  # [B, J*K]
            if row_all_masked.any():
                # 用“最近的 key”兜底：选该行距离最近（dist2 最小）的列
                nearest = dist2.view(B, J*self.kj, L*self.kl).argmin(dim=-1)  # [B, J*K]
                b_idx, r_idx = torch.where(row_all_masked)
                m[b_idx, r_idx, nearest[b_idx, r_idx]] = True

            # ——[NEW]：不再用 -inf，改用大负数，避免 softmax NaN——
            attn_bias = pos_bias.masked_fill(~m, -1e4)
        else:
            # 无限制：仅用位置偏置
            attn_bias = pos_bias


        # —— 一次性把 3D 加性 mask 喂给 MHA（如果你的 PyTorch 支持）——
        # 注意：某些版本要求 attn_mask.dtype=torch.float32 并且与输入在同一设备
        outs = []
    
        for b in range(B):
            out_b, _ = self.attn_jq_lkv(
                tj[b:b+1], tl[b:b+1], tl[b:b+1],
                attn_mask=attn_bias[b],   # 现在是浮点加性 mask，已稳
                key_padding_mask=None,
                need_weights=False
            )
            outs.append(out_b)

        out_j = torch.cat(outs, dim=0)  # [B, J*Kj, d_model]

        tj = tj + out_j  # 残差

        # ====================token -> 标量增量，再按写回============================
        ## 1) 得到增强量（可选：softplus 只增 / 线性可增可减）
        #delta_tok = F.softplus(self.proj_out(tj)).view(B, J, self.kj, 1)  # 或者：self.proj_out(tj)

        # 2) warmup 与步长
        #warm = torch.clamp(self._global_step.float() / float(self.delta_warm_iters), 0.0, 1.0) if self.training else joint_hm.new_tensor(1.0)
        #eff = self.delta_scale * warm

        # 3) 只改“值”，坐标用原来的 ij，不做位移
        #v_new = vj + eff * delta_tok                 # [B, J, K, 1]

        # 4) 原位覆盖写回（不扩散、不相加）
        #flat = joint_hm.view(B, J, -1).clone()       # [B, J, D*H*W]
        #flat.scatter_(dim=2, index=ij, src=v_new.squeeze(-1))  # 仅覆盖 top-K 位置
        #fused_joint = flat.view(B, J, D, H, W)
        #====================================================================================================

        # ====================增强后的 k 个热力值写回============================
        #方法1.直接写会
        #v_hat = torch.sigmoid(self.value_head(tj))     # 直接预测增强后的热力值 v′∈(0,1)
        #v_hat = v_hat.view(B, J, self.kj, 1)  
        # 再按原索引 ij 原位覆盖（坐标不变）
        #flat = joint_hm.view(B, J, -1).clone()
        #flat.scatter_(dim=2, index=ij, src=v_hat.squeeze(-1))
        #fused_joint = flat.view(B, J, D, H, W)

        #方法2.防止过猛写法：
        v_hat = torch.sigmoid(self.value_head(tj)).view(B, J, self.kj, 1)

        # 门控(0~1)，控制“增强幅度”（也可乘上 warmup）
        alpha = torch.sigmoid(self.gate_head(tj)).view(B, J, self.kj, 1)  # 需要在 __init__ 里定义 gate_head
        # alpha = alpha * warm  # 如果想叠加你已有的 warmup

        v_new = (1 - alpha) * vj + alpha * v_hat                           # [B, J, K, 1]

        flat = joint_hm.view(B, J, -1).clone()
        flat.scatter_(2, ij, v_new.squeeze(-1))
        fused_joint = flat.view(B, J, D, H, W)
        #====================================================================================================

        #fused_joint = joint_hm + self._scatter_writeback(dj, ij, (D, H, W), radius=self.diffusion_radius)
        # 在 forward() 内合适的位置（比如 attn_bias 构造后、写回前后）
        #if self.training:  # 只在训练时监控
        #    self._dbg_counter += 1
        #    if self._dbg_counter % 500 == 0: 
        #        def _stat(name, x):
        #            x_det = x.detach()
        #            print(f"[DBG] {name}: shape={tuple(x_det.shape)} "
        #                f"min={x_det.min().item():.4g} max={x_det.max().item():.4g} "
        #                f"mean={x_det.mean().item():.4g} std={x_det.std().item():.4g} "
        #                f"nan={torch.isnan(x_det).any().item()} inf={torch.isinf(x_det).any().item()}")

        #        _stat("joint_hm", joint_hm)
        #        _stat("limb_hm", limb_hm)
        #        _stat("attn_bias", attn_bias)           # 位置偏置（含邻接 -inf 后）
        #        _stat("tj", tj)                          # 送入注意力的关节token
        #        _stat("tl", tl)                          # 送入注意力的limb token
        #        _stat("out_j", out_j)                    # 注意力输出
                #_stat("dj (proj_out)", dj)               # 写回前的标量增量
                #delta = self._scatter_writeback(dj, ij, (D,H,W), radius=self.diffusion_radius)
                #_stat("delta", delta)                    # 写回的增量图
                # = joint_hm + getattr(self, "delta_scale", torch.tensor(1., device=joint_hm.device)) * delta
                #_stat("fused_joint", fused_joint)        # 融合后的热力图

        #        if hasattr(self, "dist_scale"):
        #            print(f"[DBG] dist_scale={float(self.dist_scale.detach()):.4g}")
        #        if hasattr(self, "delta_scale"):
        #            print(f"[DBG] delta_scale={float(self.delta_scale.detach()):.4g}")

        return fused_joint



@HEADS.register_module()
class TopKCrossattenyionHeatmap3DNet_v4nolimb(nn.Module):
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
        super(TopKCrossattenyionHeatmap3DNet_v4nolimb, self).__init__()

    
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
        

        self.limb_loss_weight = 0  # 🔧 可以调节，例如 0.1 ~ 1.0 之间  0.5

        #方法一第一阶段
        #self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)    #自动学习权重比例

        #方法一第三阶段加入编码器
        #self.joint_encoder = JointHeatmapEncoder(in_channels=1, feature_dim=128)
        #self.limb_encoder = LimbHeatmapEncoder(in_channels=1, feature_dim=128)
        #self.selector = DynamicLimbSelector(C=128, prior_dim=0, use_prior=False)
        #self.gate = nn.Linear(128, 128)
        #nn.init.constant_(self.gate.bias, -2.0)  # 初始更保守
        #self.voxel_decoder = JointVoxelDecoder(feature_dim=128, output_shape=(64, 64, 64))

        # === time ===
        #self.enable_timing = True   # 关/开计时打印
        #self.profile_every_n = 50   # 每 N 次 forward 打印一次
        #self._iter = 0              # 内部计数器


        


        self.parent_map = {
            1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
            7: 1, 8: 7, 9: 8, 10: 9,
            11: 4, 12: 11, 13: 12, 14: 13
        }
        self.LIMB_CONNECTIONS = LIMB_CONNECTIONS
        self.limb_map = {tuple(sorted((a, b))): i for i, (a, b) in enumerate(self.LIMB_CONNECTIONS)}

        self.debug_iter_counter = 0                   #测试debug
        # === FeatureMixLimbHeatmap3DNet.__init__ ===gpt5建议
        #self.alpha_param = nn.Parameter(torch.tensor(-2.0))  # 初始 alpha ~ 0.06
        # ====================================================

        #===============topk-crossattention=======
        # === TGFI-TopK 相关 ===
        cross_bidirectional   = kwargs.get('cross_bidirectional', True)
        restrict_to_adjacent  = kwargs.get('restrict_to_adjacent', True)
        diffusion_radius      = kwargs.get('diffusion_radius', 1)
        self.use_tgfi_topk = kwargs.get('use_tgfi_topk', True)
        self.zero_limb_for_cross = kwargs.get('zero_limb_for_cross', False)
        self.k_joint = kwargs.get('k_joint', 45)
        self.k_limb  = kwargs.get('k_limb', 45)
        self.cross_tgfi = CrossTGFI3D(
            d_model=64, nhead=4, k_joint=self.k_joint, k_limb=self.k_limb,
            bidirectional=cross_bidirectional,
            restrict_to_adjacent=restrict_to_adjacent,
            diffusion_radius=diffusion_radius,
            zero_limb_for_cross=self.zero_limb_for_cross,
        )
        # 为相邻约束准备关节-肢体邻接掩码
        J = self.joint_num
        L = len(self.LIMB_CONNECTIONS) if hasattr(self, 'LIMB_CONNECTIONS') else len(LIMB_CONNECTIONS)
        adj = torch.zeros(J, L, dtype=torch.bool)
        for l_id, (a, b) in enumerate(self.LIMB_CONNECTIONS):
            adj[a, l_id] = True;  adj[b, l_id] = True
        self.register_buffer("joint_to_limb_mask", adj, persistent=False)
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
        # self.limb_head = nn.Sequential(
        #     nn.Conv3d(joint_num, 64, kernel_size=3, padding=1),  # 空间结构建模
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(64, len(LIMB_CONNECTIONS), kernel_size=1)  # 输出 limb heatmap
        #         )


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

        # === timing setup ===
        #t = {}  # 保存各阶段耗时（毫秒）
        #def _sync():
        #    if torch.cuda.is_available():
        #        torch.cuda.synchronize()
        #def _tick(name, t0):
        #    _sync()
        #    t[name] = (time.perf_counter() - t0) * 1000.0  # ms
        #    return time.perf_counter()
        #======
        #_sync(); t0 = time.perf_counter()

        #======用于查看运行时 K/扩散/暖启动参数到底是多少=========
        ct = self.cross_tgfi
        import logging; logging.getLogger().info(
            f"[SANITY][TGFI] KJ={ct.kj} KL={ct.kl} "
            f"diffR={ct.diffusion_radius} "
            f"dist_scale={float(ct.dist_scale.detach()):.3f} "
            f"delta_warm_iters={ct.delta_warm_iters} "
            f"restrict={ct.restrict}"
        )
        #===================================



        img_feat = self._transform_inputs(img_feat)                        #没有进行任何操作，直接返回输入
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])
        #self.deconv(img_feat) 是调用 self.deconv 模块，将特征图上采样成 3D 热力图，输出是 [B, 15 * 64, 64, 64]
        #_t0 = _tick('deconv', t0)

        #view(-1, self.joint_num, self.heatmap_shape[0], self.heatmap_shape[1], self.heatmap_shape[2]) 是将输出展平为 [B, 15, 64, 64, 64]
        #joint_num=15, heatmap_shape=(64, 64, 64)
        

        #新加入的用来计算热力图的。这个热力图是在已有 joint 表达的结构空间中，学习 limb 表达。
        #limb_heatmap = self.limb_head(joint_3d_heatmap)
        #limb_heatmap = self.limb_head(joint_3d_heatmap.detach())        #改过
        #t0 = _tick('limb_head', t0)

        limb_for_cross = torch.zeros_like(joint_3d_heatmap)

        # if self.zero_limb_for_cross:
        #     limb_for_cross = limb_heatmap.detach() * 0.0
        # else:
        #     print('zero_limb_for_cross is not 0000')
        #     limb_for_cross = limb_heatmap

        if self.use_tgfi_topk:
            fused_joint_hm = self.cross_tgfi(
                joint_3d_heatmap,
                limb_for_cross,
                joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device)
            )
        else:
            print('use_tgfi_topk is false')
            fused_joint_hm = joint_3d_heatmap

        
        #############################测试
        #print("✅ limb_heatmap max:", limb_heatmap.max().item())
        #print("✅ limb_heatmap nonzero:", (limb_heatmap > 0).sum().item())

        #############################尝试加入特征融合
        #if self.use_tgfi_topk:
        #    fused_joint_hm = self.cross_tgfi(joint_3d_heatmap, limb_heatmap,joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device))
        #else:
        #    print('use_tgfi_topk is false')
        #    fused_joint_hm = joint_3d_heatmap
            
        joint_coord = soft_argmax_3d(fused_joint_hm)



        
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

        # =============打印一次 time（按频率）=======
        #if self.enable_timing:
        #    self._iter += 1
        #    if (self._iter % self.profile_every_n) == 0:
        #        mem = None
        #        if torch.cuda.is_available():
        #            mem = round(torch.cuda.memory_allocated() / 1e9, 3)
        #        print('[PROFILE] iter=%d ms=' % self._iter, t, 'cuda_mem(GB)=', mem)


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
        #limb_heatmap = self.limb_head(joint_3d_heatmap) 
        #limb_for_cross = limb_heatmap.detach() * 0.0
        limb_for_cross = torch.zeros_like(joint_3d_heatmap)


        fused_joint_hm = self.cross_tgfi(
                joint_3d_heatmap,
                limb_for_cross,
                joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device)
            )

        #方法三
        # == 与 forward 同步：先算 limb，再两路编码 ============================================gpt5，融合模块推添加
        ##limb_heatmap = self.limb_head(joint_3d_heatmap)
        #limb_heatmap = self.limb_head(joint_3d_heatmap)
        #fused_joint_hm = self.cross_tgfi(joint_3d_heatmap, limb_heatmap,joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device))

        joint_coord = soft_argmax_3d(fused_joint_hm)



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
        '''
        # --- 轻量：就地初始化一个计数器（如果没有） ---
        if not hasattr(self, "_save_counter"):
            self._save_counter = 0

        def _make_sample_tag(meta: dict, fallback_idx: int, counter: int) -> str:
            for k in ("image_file", "filename", "img_path", "ori_filename"):
                if isinstance(meta, dict) and k in meta and meta[k]:
                    base = os.path.basename(meta[k])
                    return os.path.splitext(base)[0]
            return f"sample_{counter:07d}_{fallback_idx:02d}"

        # ====== 保存 limb 体素热力图（pred + gt）======
        if self.test_cfg.get('save_limb_heatmap', False):
            save_dir = self.test_cfg.get('limb_heatmap_save_dir', './outputs/limb_heatmaps')
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                limb_heatmap = self.limb_head(joint_3d_heatmap)  # [B, L, D, H, W]
            limb_pred = limb_heatmap.detach().cpu().numpy()

            img_metas = kwargs.get('img_metas', None)
            gt_limb = None
            if 'target_limb_heatmap' in kwargs and kwargs['target_limb_heatmap'] is not None:
                gt_limb = kwargs['target_limb_heatmap'].detach().cpu().numpy()

            B = limb_pred.shape[0]
            for i in range(B):
                tag = _make_sample_tag(img_metas[i] if img_metas else {}, i, self._save_counter)
                np.save(os.path.join(save_dir, f'{tag}_pred_limb.npy'), limb_pred[i], allow_pickle=False)
                if gt_limb is not None:
                    np.save(os.path.join(save_dir, f'{tag}_gt_limb.npy'), gt_limb[i], allow_pickle=False)
            self._save_counter += B
            print(f"✅ Saved limb heatmaps (.npy) for {B} samples to {save_dir}")

        # ====== 保存关节 3D 热力图（反卷积输出）======
        if self.test_cfg.get('save_joint_heatmap', False):
            jhm_dir = self.test_cfg.get('joint_heatmap_save_dir', './outputs/joint_heatmaps')
            os.makedirs(jhm_dir, exist_ok=True)

            jhm_np = joint_3d_heatmap.detach().cpu().numpy()  # [B, J, D, H, W]
            img_metas = kwargs.get('img_metas', None)

            B = jhm_np.shape[0]
            for i in range(B):
                tag = _make_sample_tag(img_metas[i] if img_metas else {}, i, self._save_counter)
                np.save(os.path.join(jhm_dir, f'{tag}_joint3d.npy'), jhm_np[i], allow_pickle=False)
            self._save_counter += B
            print(f"✅ Saved joint 3D heatmaps (.npy) for {B} samples to {jhm_dir}")
        '''


        '''
        # --- 轻量：就地初始化一个计数器（如果没有） ---
        if not hasattr(self, "_save_counter"):
            self._save_counter = 0

        def _make_sample_tag(meta: dict, fallback_idx: int, counter: int) -> str:
            
            for k in ("image_file", "filename", "img_path", "ori_filename"):
                if isinstance(meta, dict) and k in meta and meta[k]:
                    base = os.path.basename(meta[k])
                    return os.path.splitext(base)[0]
            return f"sample_{counter:07d}_{fallback_idx:02d}"

        
        # ====== 保存 limb 体素热力图（pred + gt）======
        if self.test_cfg.get('save_limb_heatmap', False):
        
            save_dir = self.test_cfg.get('limb_heatmap_save_dir', './outputs/limb_heatmaps')
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                limb_heatmap = self.limb_head(joint_3d_heatmap)  # [B, L, D, H, W]
            limb_pred = limb_heatmap.detach().cpu().numpy()

            img_metas = kwargs.get('img_metas', None)
            gt_limb = None
            if 'target_limb_heatmap' in kwargs and kwargs['target_limb_heatmap'] is not None:
                gt_limb = kwargs['target_limb_heatmap'].detach().cpu().numpy()

            B = limb_pred.shape[0]
            for i in range(B):
                tag = _make_sample_tag(img_metas[i] if img_metas else {}, i, self._save_counter)
                np.savez_compressed(os.path.join(save_dir, f'{tag}_pred_limb.npz'), arr=limb_pred[i])
                if gt_limb is not None:
                    np.savez_compressed(os.path.join(save_dir, f'{tag}_gt_limb.npz'), arr=gt_limb[i])
            self._save_counter += B
            print(f"✅ Saved limb heatmaps for {B} samples to {save_dir}")

        # ====== 保存关节 3D 热力图（反卷积输出）======
        if self.test_cfg.get('save_joint_heatmap', False):
            
            jhm_dir = self.test_cfg.get('joint_heatmap_save_dir', './outputs/joint_heatmaps')
            os.makedirs(jhm_dir, exist_ok=True)

            jhm_np = joint_3d_heatmap.detach().cpu().numpy()  # [B, J, D, H, W]
            img_metas = kwargs.get('img_metas', None)

            B = jhm_np.shape[0]
            for i in range(B):
                tag = _make_sample_tag(img_metas[i] if img_metas else {}, i, self._save_counter)
                np.savez_compressed(os.path.join(jhm_dir, f'{tag}_joint3d.npz'), arr=jhm_np[i])
            self._save_counter += B
            print(f"✅ Saved joint 3D heatmaps for {B} samples to {jhm_dir}")
        '''



        '''

            #用来保存热力图的
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
        #    #sys.exit()
        # 在 inference_model 里，joint_3d_heatmap 已经算好
        if self.test_cfg.get('save_joint_heatmap', False):
            jhm_dir = self.test_cfg.get('joint_heatmap_save_dir', './outputs/joint_heatmaps')
            os.makedirs(jhm_dir, exist_ok=True)
            jhm_np = joint_3d_heatmap.detach().cpu().numpy()   # [B, J, D, H, W]
            # 建议按 batch 循环 + 用文件名做区分（需要从 kwargs 或外层 result 拿到 img_metas 里的 image_file）
            np.save(os.path.join(jhm_dir, f'joint3d_sample0.npy'), jhm_np[0])
        '''     
                
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
