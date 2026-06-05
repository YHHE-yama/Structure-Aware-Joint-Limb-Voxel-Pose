# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         get_dist_info)
from mmcv.utils import digit_version

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger

from mmcv.runner import Hook                                               #新增super权重衰减

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


#新增class，用于衰减骨骼方向/长度损失权重-------------------------------------------------------
class LossWeightSchedulerHook(Hook):
    """
    按对数尺度把某个损失权重从 start 衰减到 end，其余权重保持不变。
    支持 target in {'dir', 'len', 'limb'} 分别对应 lambda_dir / lambda_len / limb_loss_weight。
    """
    def __init__(self, target='dir', start=0.1, end=1e-4):
        assert target in ('dir', 'len', 'limb')
        self.target = target
        self.start = float(start)
        self.end = float(end)
        self.max_epochs = None

    def before_run(self, runner):
        self.max_epochs = runner.max_epochs

    def before_train_epoch(self, runner):
        # 对数插值：权重 = start * (end/start) ** progress
        # progress ∈ [0,1]，按 epoch 线性推进
        epoch = runner.epoch  # 0-based
        denom = max(1, (self.max_epochs - 1))
        progress = float(epoch) / float(denom)
        cur = self.start * ((self.end / self.start) ** progress)

        # 取到实际模型（去掉 DataParallel 包裹）
        model = runner.model
        module = getattr(model, 'module', model)

        # 兼容不同命名（一般是 keypoint_head）
        head = None
        for name in ('keypoint_head', 'pose_head', 'head'):
            if hasattr(module, name):
                head = getattr(module, name)
                break
        if head is None:
            # 退路：如果你的 head 就是 module 本体（自定义模型），也能跑
            head = module

        if self.target == 'dir':
            head.lambda_dir = cur
        elif self.target == 'len':
            head.lambda_len = cur
        else:
            head.limb_loss_weight = cur

        # 也可以顺便把当前值打进 logger
        runner.logger.info(f'[LossWeightScheduler] epoch={epoch} target={self.target} weight={cur:.6g}')
#---------------------------------------------------------------------------------------





def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority 如果 cfg.data 中包含 train_dataloader 字段，就用它来覆盖默认的 loader_cfg；否则就什么都不加，train_loader_cfg = loader_cfg。
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # determine whether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        #原始
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        #测试
        #find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if digit_version(mmcv.__version__) >= digit_version(
                '1.4.4') or torch.cuda.is_available():
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            warnings.warn(
                'We recommend to use MMCV >= 1.4.4 for CPU training. '
                'See https://github.com/open-mmlab/mmpose/pull/1157 for '
                'details.')

    # build runner 原始
    optimizer = build_optimizers(model, cfg.optimizer)

    # === SANITY: 打印优化器 param groups（看倍率是否生效） 微调的时候加入的============

    opt = optimizer if isinstance(optimizer, torch.optim.Optimizer) else list(optimizer.values())[0]

    # 用 model（而不是 runner）拿名字映射
    model_for_names = model.module if hasattr(model, 'module') else model
    param2name = {id(p): n for n, p in model_for_names.named_parameters()}

    for gi, g in enumerate(opt.param_groups):
        lr = g.get('lr', None)
        wd = g.get('weight_decay', None)
        rep = 'n/a'
        if len(g['params']) > 0:
            rep = param2name.get(id(g['params'][0]), 'n/a')
        logger.info(f"[SANITY][OPT] group#{gi}: lr={lr} wd={wd} example={rep}")
    # ======================================================

    #==============================================================

    
    # ==========================build runner===========================过滤掉冻结的参数
    #trainable_params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = build_optimizers({'params': trainable_params}, cfg.optimizer)
    # =========================


    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    custom_hooks_cfg = cfg.get('custom_hooks', None)
    if custom_hooks_cfg is None:
        custom_hooks_cfg = cfg.get('custom_hooks_config', None)
        if custom_hooks_cfg is not None:
            warnings.warn(
                '"custom_hooks_config" is deprecated, please use '
                '"custom_hooks" instead.', DeprecationWarning)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=custom_hooks_cfg)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        #==============================================================
        #from mmpose.apis.ckpt_report import summarize_ckpt_vs_model

        #ckpt = torch.load(cfg.load_from, map_location='cpu')
        #state = ckpt.get('state_dict', ckpt)

        # 如果担心把 A 的相机采样网格带到 B，建议忽略它（安全做法）
        #state.pop('backbone.fisheye2sphere.patches_2d', None)

        #model_obj = runner.model.module if hasattr(runner.model, 'module') else runner.model
        #==========================微调的时候加入的====================================
        #try:
        #    ct = model_obj.keypoint_head.cross_tgfi
        #    if hasattr(ct, "_global_step"):
        #        ct._global_step.zero_()
        #        logger.info("[SANITY][TGFI] _global_step reset to 0 after load_from")
        #except Exception as e:
        #    logger.info(f"[SANITY][TGFI] reset step skipped: {e}")
        #==============================================================

        #summarize_ckpt_vs_model(
        #    model_obj,
        #    state,
        #    #ignore_keys={'backbone.fisheye2sphere.patches_2d'},  # 可留可去
        #    group_depth=2,
        #    print_topn=20
        #)
        runner.load_checkpoint(cfg.load_from)

    #新增class，用于衰减骨骼方向/长度损失权重-------------------------------------------------------
    # 假设 cfg 里有 loss_weight_schedule 段
    lws = cfg.get('loss_weight_schedule', None)
    if lws is not None and lws.get('enabled', False):
        hook = LossWeightSchedulerHook(
            target=lws.get('target', 'dir'),
            start=lws.get('start', 0.1),
            end=lws.get('end', 1e-4),
        )
        runner.register_hook(hook, priority='NORMAL')
    #---------------------------------------------------------------------------------------

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)################################训练起点
