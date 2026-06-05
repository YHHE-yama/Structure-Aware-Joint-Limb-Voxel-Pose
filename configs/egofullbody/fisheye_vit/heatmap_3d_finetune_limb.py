_base_ = ['path to /configs/_base_/default_runtime.py']

work_dir='/path to save/'

optimizer = dict(
    type='AdamW', 
    lr=3e-5,  
    weight_decay=0.01
)


optimizer_config = None

optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[35, 40])

evaluation = dict(
    res_folder='/the path same as work_dir/',
    metric='mpjpe',
    save_best='mpjpe',
    rule='less'
)
checkpoint_config = dict(interval=3)

total_epochs = 80 
img_res = 256
fisheye_camera_path =  '/path to mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json' 
load_from = '/path to pretrained_models/vit_pretrained_model.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='simple_add_ray', name='try1')),
    ])


channel_cfg = dict(
    num_output_channels=15,
    dataset_joints=15,
    dataset_channel=[
        list(range(15)),
    ],
    inference_channel=list(range(15)))

# model settings
model = dict(
    type='Egocentric3DPoseEstimator',
    pretrained='path to pretrained_models/SceneEgo_limb_pretrain.pth',
    backbone=dict(
        type='UndistortViT',
        img_size=(img_res, img_res),
        patch_size=16,
        embed_dim=768,                                                       
        depth=12,                                                            
        num_heads=12,                                                        
        ratio=1,
        use_checkpoint=False,                                                
        mlp_ratio=4,                                                         
        qkv_bias=True,                                                       
        drop_path_rate=0.3,                                                  
        fisheye2sphere_configs=dict(
            type='UndistortPatch',                                           
            input_feature_height=256,
            input_feature_width=256,                                         
            image_h=1024,                                                    
            image_w=1280,                                                    
            patch_num_horizontal=16,
            patch_num_vertical=16,                                           
            patch_size=(0.1, 0.1),                                           
            patch_pixel_number=(16, 16),                                     
            crop_to_square=True,                                             
            camera_param_path=fisheye_camera_path,                           
        )
    ),
    keypoint_head=dict(                                                      
        type='TopKCrossattenyionHeatmap3DNet_v2',
        #type='TopKCrossattenyionHeatmap3DNet_v3',
        #type='TopKCrossattenyionHeatmap3DNet_v4nolimb',                          
        in_channels=768,                                                     
        num_deconv_layers=2,                                                 
        num_deconv_filters=(1024, 15 * 64),                                  
        num_deconv_kernels=(4, 4),                                           
        out_channels=15 * 64,                                                
        heatmap_shape=(64, 64, 64),                                          
        fisheye_model_path=fisheye_camera_path, joint_num=15,
        loss_keypoint=dict(type='MPJPELoss', use_target_weight=True),         
        ##================topk-attention==========
        use_tgfi_topk=True,                   
        k_joint=30,
        k_limb=30,
        cross_bidirectional=False,     
        restrict_to_adjacent=True,    
        diffusion_radius=0,           
        #zero_limb_for_cross=True,             #for v4nolimb
        
    
    ),
    train_cfg=dict(),
    test_cfg=dict()
)

data_cfg = dict(
    num_joints=15,
    camera_param_path=fisheye_camera_path,
    joint_type='mo2cap2',
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    joint_weights=[1.] * 15,
    use_different_joint_weights=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img', 'keypoints_3d', 'keypoints_3d_visible','target_limb_heatmap'
        ],
        meta_keys=['image_file', 'joints_2d']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='Generate2DPoseConfidence'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],#'target_limb_heatmap',
        meta_keys=['image_file', 'keypoints_3d', 'keypoints_3d_visible']),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,  
    val_dataloader=dict(samples_per_gpu=10, workers_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(samples_per_gpu=10, workers_per_gpu=1, persistent_workers=False),
    train=dict(
        type='LimbMocapStudioFinetuneDataset',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='MocapStudioDataset', 
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type='MocapStudioDataset',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
