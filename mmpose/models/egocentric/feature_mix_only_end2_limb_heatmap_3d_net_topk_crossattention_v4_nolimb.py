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
from .voxel_fusion_modules import JointHeatmapEncoder, LimbHeatmapEncoder, DynamicLimbSelector,JointVoxelDecoder 

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
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))    
    heatmap3d = F.softmax(heatmap3d, 2)                                        
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))      

    accu_x = heatmap3d.sum(dim=(2, 3))                                        
    accu_y = heatmap3d.sum(dim=(2, 4))                                        
    accu_z = heatmap3d.sum(dim=(3, 4))                                        
    
    device = heatmap3d.device

    accu_x = accu_x * torch.arange(width, device=device).float()[None, None, :]
    accu_y = accu_y * torch.arange(height, device=device).float()[None, None, :]
    accu_z = accu_z * torch.arange(depth, device=device).float()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


class CrossTGFI3D(nn.Module):

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

        self.embed = nn.Linear(1 + 3, d_model)
        self.proj_out = nn.Linear(d_model, 1)  
        self.value_head = nn.Linear(d_model, 1) 
        self.gate_head  = nn.Linear(d_model, 1)

        self.attn_jq_lkv = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        if bidirectional:
            self.attn_lq_jkv = nn.MultiheadAttention(d_model, nhead, batch_first=True)


        self.dist_scale = nn.Parameter(torch.tensor(2.0))
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self._dbg_counter = 0
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long))
        self.delta_warm_iters = 1000

    @staticmethod
    def _topk_tokenize_3d(x, k):

        B, C, D, H, W = x.shape
        N = D * H * W
        flat = x.view(B, C, N)
        val, idx = torch.topk(flat, min(k, N), dim=-1)  
        z = torch.div(idx, H * W, rounding_mode='floor').float()
        y = torch.div(idx.remainder(H * W), W, rounding_mode='floor').float()
        x_ = idx.remainder(W).float()

        pos = torch.stack([x_ / (W - 1 + 1e-6),
                           y / (H - 1 + 1e-6),
                           z / (D - 1 + 1e-6)], dim=-1)  # [B,C,K,3]
        return val.unsqueeze(-1), idx, pos  


    @staticmethod
    def _scatter_writeback(delta_tokens, idx, shape, radius=1):

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

        P2 = (P * P).sum(-1, keepdim=True)          
        Q2 = (Q * Q).sum(-1, keepdim=True).transpose(1, 2)  
        dist2 = P2 + Q2 - 2.0 * P @ Q.transpose(1, 2)
        return dist2.clamp_min_(0.0)

    def forward(self, joint_hm, limb_hm, joint_to_limb_mask=None):

        B, J, D, H, W = joint_hm.shape
        L = limb_hm.shape[1]

        vj, ij, pj = self._topk_tokenize_3d(joint_hm, self.kj)  
        vl, il, pl = self._topk_tokenize_3d(limb_hm, self.kl)


        tj = self.embed(torch.cat([vj, pj], dim=-1)).view(B, J*self.kj, self.d_model)
        tl = self.embed(torch.cat([vl, pl], dim=-1)).view(B, L*self.kl, self.d_model)
        
        if self.zero_limb_for_cross:
            tl = tl.detach() * 0.0   
        else:
            print('zero_limb_for_cross is not 0000')
            tl = tl

        Pj = pj.view(B, J*self.kj, 3)
        Pl = pl.view(B, L*self.kl, 3)
        dist2 = self._pairwise_dist(Pj, Pl)  

        dist_scale = F.softplus(self.dist_scale)  
        pos_bias   = -dist2 * dist_scale          

        if self.restrict and (joint_to_limb_mask is not None):

            m = joint_to_limb_mask[None, :, None, :, None].to(torch.bool).to(tj.device)  
            m = m.expand(B, J, self.kj, L, self.kl).reshape(B, J*self.kj, L*self.kl)

            row_all_masked = ~m.any(dim=-1)                  
            if row_all_masked.any():

                nearest = dist2.view(B, J*self.kj, L*self.kl).argmin(dim=-1)  
                b_idx, r_idx = torch.where(row_all_masked)
                m[b_idx, r_idx, nearest[b_idx, r_idx]] = True

            attn_bias = pos_bias.masked_fill(~m, -1e4)
        else:

            attn_bias = pos_bias

        outs = []
    
        for b in range(B):
            out_b, _ = self.attn_jq_lkv(
                tj[b:b+1], tl[b:b+1], tl[b:b+1],
                attn_mask=attn_bias[b],   
                key_padding_mask=None,
                need_weights=False
            )
            outs.append(out_b)

        out_j = torch.cat(outs, dim=0)  

        tj = tj + out_j  

        v_hat = torch.sigmoid(self.value_head(tj)).view(B, J, self.kj, 1)

        alpha = torch.sigmoid(self.gate_head(tj)).view(B, J, self.kj, 1)  

        v_new = (1 - alpha) * vj + alpha * v_hat                           

        flat = joint_hm.view(B, J, -1).clone()
        flat.scatter_(2, ij, v_new.squeeze(-1))
        fused_joint = flat.view(B, J, D, H, W)
       
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
        self.final_conv = nn.Conv2d(num_deconv_filters[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.heatmap_shape = heatmap_shape
        self.fisheye_model = FishEyeCameraCalibrated(fisheye_model_path)
        self.joint_num = joint_num
        self.voxel_size = voxel_size
        self.loss = build_loss(loss_keypoint)
        

        self.limb_loss_weight = 0  

        self.parent_map = {
            1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5,
            7: 1, 8: 7, 9: 8, 10: 9,
            11: 4, 12: 11, 13: 12, 14: 13
        }
        self.LIMB_CONNECTIONS = LIMB_CONNECTIONS
        self.limb_map = {tuple(sorted((a, b))): i for i, (a, b) in enumerate(self.LIMB_CONNECTIONS)}

        self.debug_iter_counter = 0                   

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

        J = self.joint_num
        L = len(self.LIMB_CONNECTIONS) if hasattr(self, 'LIMB_CONNECTIONS') else len(LIMB_CONNECTIONS)
        adj = torch.zeros(J, L, dtype=torch.bool)
        for l_id, (a, b) in enumerate(self.LIMB_CONNECTIONS):
            adj[a, l_id] = True;  adj[b, l_id] = True
        self.register_buffer("joint_to_limb_mask", adj, persistent=False)

        self.input_transform = input_transform
        self.in_index = in_index
        logger = logging.getLogger()
        logger.info(f'Input transform: {self.input_transform}')
        logger.info(f'Input index: {self.in_index}')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



    def _transform_inputs(self, inputs):

        if not isinstance(inputs, list) or self.input_transform is None:
            return inputs                                                   

        if self.input_transform == 'resize_concat':                         
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
            
            B, C, H, W = inputs.shape
            if H != 16 or W != 16:
                
                inputs = F.interpolate(inputs, size=(16, 16), mode='bilinear', align_corners=False)
        elif self.input_transform == 'multiple_select':                   
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]                                

        return inputs
    
    def _get_parent_feature(self, joint_feat, j):

        if j in self.parent_map:
            parent_j = self.parent_map[j]
        else:
            parent_j = j
            
        return joint_feat[:, parent_j], parent_j
        
    def find_limb_index(self, parent, child):
        for i, (a, b) in enumerate(self.LIMB_CONNECTIONS):  
            if (a == parent and b == child) or (a == child and b == parent):
                return i
        return None  


    def forward(self, img_feat):


        ct = self.cross_tgfi
        import logging; logging.getLogger().info(
            f"[SANITY][TGFI] KJ={ct.kj} KL={ct.kl} "
            f"diffR={ct.diffusion_radius} "
            f"dist_scale={float(ct.dist_scale.detach()):.3f} "
            f"delta_warm_iters={ct.delta_warm_iters} "
            f"restrict={ct.restrict}"
        )



        img_feat = self._transform_inputs(img_feat)                        
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])

        limb_for_cross = torch.zeros_like(joint_3d_heatmap)

        if self.use_tgfi_topk:
            fused_joint_hm = self.cross_tgfi(
                joint_3d_heatmap,
                limb_for_cross,
                joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device)
            )
        else:
            print('use_tgfi_topk is false')
            fused_joint_hm = joint_3d_heatmap
            
        joint_coord = soft_argmax_3d(fused_joint_hm)

        resize_z = self.voxel_size[2] / self.heatmap_shape[2]

        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z

        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024


        joint_coord = self.fisheye2camera(joint_coord)       

        return {
                'preds': joint_coord,
               }

    def fisheye2camera(self, joint_coord):

        batch_size = joint_coord.shape[0]
        joint_coord_xy = joint_coord[:, :, :2].view(batch_size * self.joint_num, 2)
        joint_coord_z = joint_coord[:, :, 2].view(batch_size * self.joint_num)
        joint_coord_cam = self.fisheye_model.camera2world_pytorch(joint_coord_xy, joint_coord_z)
        joint_coord_cam = joint_coord_cam.view(batch_size, self.joint_num, 3)
        joint_coord_cam = joint_coord_cam.contiguous()
        return joint_coord_cam

    def get_loss(self, output, keypoints_3d, keypoint_3d_visible=None, **kwargs):

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert keypoints_3d.dim() == 3
        losses['mpjpe_loss'] = self.loss(output['preds'], keypoints_3d, keypoint_3d_visible[:, :, None])

        pred, gt = output['preds'], keypoints_3d                  
        angle_loss, len_loss = 0., 0.
        for (a, b) in LIMB_CONNECTIONS:
            v_pred = pred[:, b] - pred[:, a]
            v_gt   = gt[:, b]   - gt[:, a]
            angle_loss += (1 - F.cosine_similarity(v_pred, v_gt, dim=-1)).mean()
            len_loss   += F.l1_loss(v_pred.norm(dim=-1), v_gt.norm(dim=-1))
        losses['bone_dir'] = 1e-3 * angle_loss
        losses['bone_len'] = 1e-3 * len_loss

        if 'limb_heatmap' in output and 'target_limb_heatmap' in kwargs:
            pred_limb = output['limb_heatmap']
            target_limb = kwargs['target_limb_heatmap']


            limb_loss = F.mse_loss(pred_limb, target_limb)

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
                    #print("🔍 limb_heatmap ：")
                    #print(corr.cpu().numpy())
                self._debug_printed = True



            losses['limb_loss'] = self.limb_loss_weight * limb_loss

        return losses

    def get_accuracy(self, output, keypoints_3d, keypoint_3d_visible=None):

        accuracy = dict()
        pred = output['preds']
        N, K, _ = pred.shape

        mpjpe = keypoint_mpjpe(
            pred.detach().cpu().numpy(),
            keypoints_3d.detach().cpu().numpy(),
            mask=keypoint_3d_visible.detach().cpu().numpy().astype(np.bool), alignment='none')
        accuracy['mpjpe'] = float(mpjpe)

        return accuracy

    def inference_model(self, x, flip_pairs=None,**kwargs):  

        img_feat = self._transform_inputs(x)
        joint_3d_heatmap = self.deconv(img_feat).view(-1, self.joint_num, self.heatmap_shape[0],
                                                      self.heatmap_shape[1], self.heatmap_shape[2])

        limb_for_cross = torch.zeros_like(joint_3d_heatmap)


        fused_joint_hm = self.cross_tgfi(
                joint_3d_heatmap,
                limb_for_cross,
                joint_to_limb_mask=joint_to_limb_mask.to(joint_3d_heatmap.device)
            )


        joint_coord = soft_argmax_3d(fused_joint_hm)



        joint_coord_in_voxel = joint_coord.detach().clone()

        resize_z = self.voxel_size[2] / self.heatmap_shape[2]

        joint_coord[:, :, 2] = joint_coord[:, :, 2] * resize_z

        joint_coord[:, :, 0] = joint_coord[:, :, 0] / self.heatmap_shape[0] * 1024 + 128
        joint_coord[:, :, 1] = joint_coord[:, :, 1] / self.heatmap_shape[1] * 1024


        joint_coord = self.fisheye2camera(joint_coord)
        result = {'keypoints_pred': joint_coord.detach().cpu().numpy()}
        if 'return_heatmap' in self.test_cfg and self.test_cfg['return_heatmap'] is True:
            result['heatmap'] = joint_3d_heatmap.detach().cpu().numpy()

        if 'return_confidence' in self.test_cfg and self.test_cfg['return_confidence'] is True:

            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_voxel = joint_coord_in_voxel.detach()
            batch_size, joint_num, depth, height, width = joint_3d_heatmap_confidence.shape

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

        if 'return_2d_heatmap' in self.test_cfg and self.test_cfg['return_2d_heatmap'] is True:

            joint_3d_heatmap_confidence = joint_3d_heatmap.detach()
            joint_3d_heatmap_confidence = joint_3d_heatmap_confidence.cpu().numpy()
            joint_3d_heatmap_confidence = gaussian_filter(joint_3d_heatmap_confidence, sigma=self.test_cfg['sigma'],
                                                          axes=(2, 3, 4))

            joint_2d_heatmap_confidence = joint_3d_heatmap_confidence.sum(axis=2)
            result['heatmap_2d'] = joint_2d_heatmap_confidence


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
                    dict(type='deconv'),     
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
