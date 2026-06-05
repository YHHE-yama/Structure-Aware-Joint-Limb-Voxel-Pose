#  Copyright Jian Wang @ MPI-INF (c) 2023.
import warnings
import torch

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from mmpose.models.detectors.base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class Egocentric3DPoseEstimator(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None,
                 freeze_backbone=False
                 ):
        super().__init__()
        self.fp16_enabled = False

        self.backbone = builder.build_backbone(backbone)     #相当于self.backbone = UndistortViT(**cfg['backbone_without_type'])，相当于是构建了undistort_vit模型，执行了__init__函数
        self.freeze_backbone = freeze_backbone

        self.train_cfg = train_cfg                        #这两条是用来添加训练和测试的配置，比如开启返回热力图功能
        self.test_cfg = test_cfg

        if neck is not None:                              #在作者的config中没有构建neck，所以这里没有执行
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg         #如果有用到keypoint_head，则将train_cfg和test_cfg传入。相当于给keypoint_head指令使用一些功能。
            keypoint_head['test_cfg'] = test_cfg

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for TopDown is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose       #这里是为了兼容之前的版本，之前用的是loss_pose，现在用的是loss_keypoint

            self.keypoint_head = builder.build_head(keypoint_head)   #相当于self.keypoint_head = build_from_cfg(**cfg['keypoint_head_without_type'])
        self.pretrained = pretrained
        self.init_weights(self.pretrained)

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    #原始的权重设置
    '''
    def init_weights(self, pretrained):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
    '''
    def init_weights(self, pretrained):
        if pretrained is not None:
            # 读取 checkpoint
            checkpoint = torch.load(pretrained, map_location='cpu')

            # 如果是 mmpose 保存的模型
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # 只保留你列出的 keys
            allowed_keys = [
                'backbone.pos_embed', 
                'backbone.fisheye2sphere.patches_2d',
                'backbone.patch_embed.proj.weight',
                'backbone.patch_embed.proj.bias',
                'backbone.blocks.0.norm1.weight',
                'backbone.blocks.0.norm1.bias',
                'backbone.blocks.0.attn.qkv.weight',
                'backbone.blocks.0.attn.qkv.bias',
                'backbone.blocks.0.attn.proj.weight',
                'backbone.blocks.0.attn.proj.bias',
                'backbone.blocks.0.norm2.weight',
                'backbone.blocks.0.norm2.bias',
                'backbone.blocks.0.mlp.fc1.weight',
                'backbone.blocks.0.mlp.fc1.bias',
                'backbone.blocks.0.mlp.fc2.weight',
                'backbone.blocks.0.mlp.fc2.bias',
                'backbone.blocks.1.norm1.weight',
                'backbone.blocks.1.norm1.bias',
                'backbone.blocks.1.attn.qkv.weight',
                'backbone.blocks.1.attn.qkv.bias',
                'backbone.blocks.1.attn.proj.weight',
                'backbone.blocks.1.attn.proj.bias',
                'backbone.blocks.1.norm2.weight',
                'backbone.blocks.1.norm2.bias',
                'backbone.blocks.1.mlp.fc1.weight',
                'backbone.blocks.1.mlp.fc1.bias',
                'backbone.blocks.1.mlp.fc2.weight',
                'backbone.blocks.1.mlp.fc2.bias',
                'backbone.blocks.2.norm1.weight',
                'backbone.blocks.2.norm1.bias',
                'backbone.blocks.2.attn.qkv.weight',
                'backbone.blocks.2.attn.qkv.bias',
                'backbone.blocks.2.attn.proj.weight',
                'backbone.blocks.2.attn.proj.bias',
                'backbone.blocks.2.norm2.weight',
                'backbone.blocks.2.norm2.bias',
                'backbone.blocks.2.mlp.fc1.weight',
                'backbone.blocks.2.mlp.fc1.bias',
                'backbone.blocks.2.mlp.fc2.weight',
                'backbone.blocks.2.mlp.fc2.bias',
                'backbone.blocks.3.norm1.weight',
                'backbone.blocks.3.norm1.bias',
                'backbone.blocks.3.attn.qkv.weight',
                'backbone.blocks.3.attn.qkv.bias',
                'backbone.blocks.3.attn.proj.weight',
                'backbone.blocks.3.attn.proj.bias',
                'backbone.blocks.3.norm2.weight',
                'backbone.blocks.3.norm2.bias',
                'backbone.blocks.3.mlp.fc1.weight',
                'backbone.blocks.3.mlp.fc1.bias',
                'backbone.blocks.3.mlp.fc2.weight',
                'backbone.blocks.3.mlp.fc2.bias',
                'backbone.blocks.4.norm1.weight',
                'backbone.blocks.4.norm1.bias',
                'backbone.blocks.4.attn.qkv.weight',
                'backbone.blocks.4.attn.qkv.bias',
                'backbone.blocks.4.attn.proj.weight',
                'backbone.blocks.4.attn.proj.bias',
                'backbone.blocks.4.norm2.weight',
                'backbone.blocks.4.norm2.bias',
                'backbone.blocks.4.mlp.fc1.weight',
                'backbone.blocks.4.mlp.fc1.bias',
                'backbone.blocks.4.mlp.fc2.weight',
                'backbone.blocks.4.mlp.fc2.bias',
                'backbone.blocks.5.norm1.weight',
                'backbone.blocks.5.norm1.bias',
                'backbone.blocks.5.attn.qkv.weight',
                'backbone.blocks.5.attn.qkv.bias',
                'backbone.blocks.5.attn.proj.weight',
                'backbone.blocks.5.attn.proj.bias',
                'backbone.blocks.5.norm2.weight',
                'backbone.blocks.5.norm2.bias',
                'backbone.blocks.5.mlp.fc1.weight',
                'backbone.blocks.5.mlp.fc1.bias',
                'backbone.blocks.5.mlp.fc2.weight',
                'backbone.blocks.5.mlp.fc2.bias',
                'backbone.blocks.6.norm1.weight',
                'backbone.blocks.6.norm1.bias',
                'backbone.blocks.6.attn.qkv.weight',
                'backbone.blocks.6.attn.qkv.bias',
                'backbone.blocks.6.attn.proj.weight',
                'backbone.blocks.6.attn.proj.bias',
                'backbone.blocks.6.norm2.weight',
                'backbone.blocks.6.norm2.bias',
                'backbone.blocks.6.mlp.fc1.weight',
                'backbone.blocks.6.mlp.fc1.bias',
                'backbone.blocks.6.mlp.fc2.weight',
                'backbone.blocks.6.mlp.fc2.bias',
                'backbone.blocks.7.norm1.weight',
                'backbone.blocks.7.norm1.bias',
                'backbone.blocks.7.attn.qkv.weight',
                'backbone.blocks.7.attn.qkv.bias',
                'backbone.blocks.7.attn.proj.weight',
                'backbone.blocks.7.attn.proj.bias',
                'backbone.blocks.7.norm2.weight',
                'backbone.blocks.7.norm2.bias',
                'backbone.blocks.7.mlp.fc1.weight',
                'backbone.blocks.7.mlp.fc1.bias',
                'backbone.blocks.7.mlp.fc2.weight',
                'backbone.blocks.7.mlp.fc2.bias',
                'backbone.blocks.8.norm1.weight',
                'backbone.blocks.8.norm1.bias',
                'backbone.blocks.8.attn.qkv.weight',
                'backbone.blocks.8.attn.qkv.bias',
                'backbone.blocks.8.attn.proj.weight',
                'backbone.blocks.8.attn.proj.bias',
                'backbone.blocks.8.norm2.weight',
                'backbone.blocks.8.norm2.bias',
                'backbone.blocks.8.mlp.fc1.weight',
                'backbone.blocks.8.mlp.fc1.bias',
                'backbone.blocks.8.mlp.fc2.weight',
                'backbone.blocks.8.mlp.fc2.bias',
                'backbone.blocks.9.norm1.weight',
                'backbone.blocks.9.norm1.bias',
                'backbone.blocks.9.attn.qkv.weight',
                'backbone.blocks.9.attn.qkv.bias',
                'backbone.blocks.9.attn.proj.weight',
                'backbone.blocks.9.attn.proj.bias',
                'backbone.blocks.9.norm2.weight',
                'backbone.blocks.9.norm2.bias',
                'backbone.blocks.9.mlp.fc1.weight',
                'backbone.blocks.9.mlp.fc1.bias',
                'backbone.blocks.9.mlp.fc2.weight',
                'backbone.blocks.9.mlp.fc2.bias',
                'backbone.blocks.10.norm1.weight',
                'backbone.blocks.10.norm1.bias',
                'backbone.blocks.10.attn.qkv.weight',
                'backbone.blocks.10.attn.qkv.bias',
                'backbone.blocks.10.attn.proj.weight',
                'backbone.blocks.10.attn.proj.bias',
                'backbone.blocks.10.norm2.weight',
                'backbone.blocks.10.norm2.bias',
                'backbone.blocks.10.mlp.fc1.weight',
                'backbone.blocks.10.mlp.fc1.bias',
                'backbone.blocks.10.mlp.fc2.weight',
                'backbone.blocks.10.mlp.fc2.bias',
                'backbone.blocks.11.norm1.weight',
                'backbone.blocks.11.norm1.bias',
                'backbone.blocks.11.attn.qkv.weight',
                'backbone.blocks.11.attn.qkv.bias',
                'backbone.blocks.11.attn.proj.weight',
                'backbone.blocks.11.attn.proj.bias',
                'backbone.blocks.11.norm2.weight',
                'backbone.blocks.11.norm2.bias',
                'backbone.blocks.11.mlp.fc1.weight',
                'backbone.blocks.11.mlp.fc1.bias',
                'backbone.blocks.11.mlp.fc2.weight',
                'backbone.blocks.11.mlp.fc2.bias',
                'backbone.last_norm.weight',
                'backbone.last_norm.bias',
                'keypoint_head.deconv.0.weight',
                'keypoint_head.deconv.1.weight',
                'keypoint_head.deconv.1.bias',
                'keypoint_head.deconv.1.running_mean',
                'keypoint_head.deconv.1.running_var',
                'keypoint_head.deconv.1.num_batches_tracked',
                'keypoint_head.deconv.3.weight',
                'keypoint_head.deconv.4.weight',
                'keypoint_head.deconv.4.bias',
                'keypoint_head.deconv.4.running_mean',
                'keypoint_head.deconv.4.running_var',
                'keypoint_head.deconv.4.num_batches_tracked',
                'keypoint_head.final_conv.weight',
                'keypoint_head.final_conv.bias',
                'keypoint_head.limb_head.0.weight',
                'keypoint_head.limb_head.0.bias',
                'keypoint_head.limb_head.1.weight',
                'keypoint_head.limb_head.1.bias',
                'keypoint_head.limb_head.1.running_mean',
                'keypoint_head.limb_head.1.running_var',
                'keypoint_head.limb_head.1.num_batches_tracked',
                'keypoint_head.limb_head.3.weight',
                'keypoint_head.limb_head.3.bias',
                'keypoint_head.limb_head.4.weight',
                'keypoint_head.limb_head.4.bias',
                'keypoint_head.limb_head.4.running_mean',
                'keypoint_head.limb_head.4.running_var',
                'keypoint_head.limb_head.4.num_batches_tracked',
                'keypoint_head.limb_head.6.weight',
                'keypoint_head.limb_head.6.bias',
            ]
            filtered_state_dict = {k: v for k, v in checkpoint.items() if k in allowed_keys}

            # ======加载到模型（只匹配已有的层）
            self.load_state_dict(filtered_state_dict, strict=False)
            #for name, param in self.named_parameters():
            #    if name in allowed_keys:   # allowed_keys 就是你上面那长串
            #        param.requires_grad = False

            for _, p in self.named_parameters():
                p.requires_grad = True

            #冻结
            #for name, p in self.named_parameters():
            #    #if name.startswith('backbone.') or name.startswith('keypoint_head.limb_head.'):
            #    if name.startswith('backbone.'):
            #        p.requires_grad = False
            #    else:
            #        p.requires_grad = True
            
            print("\n=== 冻结的层列表 ===")
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    print(f"[Frozen] {name}")
            print("====================\n")

            


        # 原有初始化
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()




    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                keypoints_3d=None,
                keypoints_3d_visible=None,
                img_metas=None,
                return_loss=True,
                return_features=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, keypoints_3d, keypoints_3d_visible, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_features=return_features, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""

        #=====原始冻结==================
        #if self.freeze_backbone:
        #    self.backbone.requires_grad = False

        #print("【Backbone 输入】img.shape:", img.shape)
        output = self.backbone(img)
        #print("【Backbone 输出】features.shape:", output.shape)
        if self.with_neck:
            output = self.neck(output)
        #    print("【Neck 输出】features.shape:", output.shape)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        #    print("【Keypoint Head 输出】output['keypoints_pred'].shape:", output['keypoints_pred'].shape)
        #    print("【Keypoint Head 输出】output['limb_heatmap'].shape:", output['limb_heatmap'].shape)  #这里是用来查看 limb heatmap 的形状
            
        #print('target_weight=======', target_weight)

        #====新增 ↓↓↓ LimbUnderlineRenderpeopleMixamoDataset====
        #if 'target_limb_heatmap' in kwargs and isinstance(kwargs['target_limb_heatmap'], torch.Tensor):
        #    tlh = kwargs['target_limb_heatmap']
        #    if tlh.ndim == 4:                      # 防止是 [L,D,H,W] 少了 batch 维
        #        tlh = tlh.unsqueeze(0)
        #    kwargs['target_limb_heatmap'] = tlh.to(
        #        device=output['limb_heatmap'].device,
        #        dtype=output['limb_heatmap'].dtype,
        #        non_blocking=True
        #    ).contiguous()
        #elif isinstance(img_metas, list) and len(img_metas) > 0 and 'target_limb_heatmap' in img_metas[0]:
            # 兼容老路径：从 meta 里取再搬设备
        #    kwargs['target_limb_heatmap'] = torch.stack(
        #        [m['target_limb_heatmap'] for m in img_metas]
        #    ).to(
        #        device=output['limb_heatmap'].device,
        #        dtype=output['limb_heatmap'].dtype,
        #        non_blocking=True
        #    ).contiguous()
        # ========================================================

        #这里加入了新东西！！！！！！！！！！！！！！！！！！最初的limbloss计算
        if isinstance(img_metas, list) and 'target_limb_heatmap' in img_metas[0]:
            kwargs['target_limb_heatmap'] = torch.stack(
                [meta['target_limb_heatmap'] for meta in img_metas]
            ).to(output['limb_heatmap'].device)
        #############################

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight,**kwargs)         #这里加入了一个,**kwargs
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, img, img_metas,return_features=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        #=====由于'img_original'报错添加的======
        allow = {'target_limb_heatmap','img_metas'}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in allow}

        #========================================

        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}
        if return_features:
        # 真的需要特征就走这个分支（如果你已有 forward_test_with_features）
            return self.forward_test_with_features(img, img_metas, **safe_kwargs)
            #return self.forward_test_with_features(img, img_metas, **kwargs)

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            
            #pred = self.keypoint_head.inference_model(features, flip_pairs=None)   #这里是原来的代码
            
            #########这里加入了新东西！！！！！！！！！！！！！！！！！！用来获取 limb heatmap 的
            pred = self.keypoint_head.inference_model(features, flip_pairs=None, **safe_kwargs)
            #pred = self.keypoint_head.inference_model(features, flip_pairs=None, **kwargs)
            #############################

            if type(pred) is dict:
                result.update(pred)
            else:
                result['keypoints_pred'] = pred
        return result

    def forward_test_with_features(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape

        result = {'img_metas': img_metas}

        features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            keypoints_pred, features_out = self.keypoint_head.forward_with_feature(features)
            result['keypoints_pred'] = keypoints_pred
            result['features'] = features_out
        return result



    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
