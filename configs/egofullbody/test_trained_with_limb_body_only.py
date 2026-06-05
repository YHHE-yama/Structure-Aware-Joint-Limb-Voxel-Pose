_base_ = ['path to /configs/_base_/default_runtime.py']


work_dir = '/path to save/'


optimizer = dict(
    type='Adam',
    lr=1e-4,
)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[40, 55])

evaluation = dict(
    metric=['mpjpe', 'pa-mpjpe']
    # save_best='pck',
    # rule='greater'
)
checkpoint_config = dict(interval=1)

total_epochs = 10
img_res = 256

fisheye_camera_path = '/path to mmpose/utils/fisheye_camera/fisheye.calibration_01_12.json'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='epoch_9')),
    ])

model = dict(
    type='Egocentric3DPoseEstimator',
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
        ##================TopK==========
        use_tgfi_topk=True,                   
        k_joint=30,
        k_limb=30,
        cross_bidirectional=False,     
        restrict_to_adjacent=True,    
        diffusion_radius=0,  
        #zero_limb_for_cross=True,             #for v4nolimb
    ),
    train_cfg=dict(),
    test_cfg=dict(
        return_heatmap=False,
        return_confidence=True,
        sigma=3,
        fisheye_camera_path=fisheye_camera_path,
    ),   
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original'],
         meta_keys=['image_file', 'keypoints_3d',
                    ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='Generate2DHandPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropHandImageFisheye', fisheye_camera_path=fisheye_camera_path,
         input_img_h=1024, input_img_w=1280,
         crop_img_size=256, enlarge_scale=1.3),
    dict(type='RGB2BGRHand'),
    dict(type='ToTensorHand'),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original', 'left_hand_img', 'right_hand_img', 'left_hand_transform', 'right_hand_transform'],
         meta_keys=['image_file', 'keypoints_3d', 'left_hand_keypoints_3d', 'right_hand_keypoints_3d', 'ext_pose_gt',
                    'ego_camera_pose', 'ext_id', 'seq_name'
                    ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropCircle', img_h=1024, img_w=1280),
    dict(type='CopyImage', source='img', target='img_original'),
    dict(type='Generate2DPose', fisheye_model_path=fisheye_camera_path),
    dict(type='CropImage', crop_left=128, crop_right=128, crop_top=0, crop_bottom=0),
    dict(type='ResizeImage', img_h=img_res, img_w=img_res),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='Collect',
         keys=['img', 'img_original'],#'target_limb_heatmap'],                                # ],# 新加入,'target_limb_heatmap'
         meta_keys=['image_file', 'keypoints_3d', 'keypoints_3d_visible']),
]

data_cfg = dict(
    num_joints=15,
    camera_param_path=fisheye_camera_path,
    joint_type='mo2cap2',
    image_size=[img_res, img_res],
    heatmap_size=(64, 64),
    #joint_weights=None,
    joint_weights=[1.] * 15,
    use_different_joint_weights=False,
)


data = dict(
    samples_per_gpu=56,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=56),
    test_dataloader=dict(samples_per_gpu=56),
    train=dict(
        type='RenderpeopleMixamoDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo/renderpeople_mixamo_labels.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type='RenderpeopleMixamoHandTestDataset',
        ann_file='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test/renderpeople_mixamo_labels_test.pkl',
        img_prefix='/HPS/ScanNet/work/synthetic_dataset_egofullbody/render_people_mixamo_test',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        part_dataset=None,
    ),
    
    #test on EgoWholeMocap test datafile
     test=dict(
         type='RenderpeopleMixamoTestDataset',                        
         ann_file='path to EgoWholeMocap/test/render_people_mixamo_test_seq/renderpeople_mixamo_labels_test_seq.pkl',
         img_prefix='path to/EgoWholeMocap/test/render_people_mixamo_test_seq',   
         data_cfg=data_cfg,
         pipeline=test_pipeline,
         #part_dataset=None,
     ),

    #test on MocapStudio test datafile
    #test=dict(
    #    type='MocapStudioDataset',                          
    #    data_cfg=data_cfg,
    #    pipeline=test_pipeline,
    #    #part_dataset=None,                        
    #),
)
