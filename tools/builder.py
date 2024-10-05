import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)

    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def get_dataset(args, config):

    dataset = build_dataset_from_cfg(config._base_, config.others)

    return dataset

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_opti_sche(base_model, config):
    # 设置优化器配置为config.optimizer
    opti_config = config.optimizer
    # 如果优化器类型为'AdamW'
    if opti_config.type == 'AdamW':
        # 定义一个函数add_weight_decay，用于添加权重衰减
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            # 遍历模型中的参数
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # 冻结的权重
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # 将不需要衰减的参数加入no_decay列表
                    no_decay.append(param)
                else:
                    # 将需要衰减的参数加入decay列表
                    decay.append(param)
            # 返回参数组，包括需要衰减和不需要衰减的参数
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        # 调用add_weight_decay函数，得到参数组
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        # 使用optim.AdamW构建优化器，传入参数组和其他参数
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    # 如果优化器类型为'Adam'
    elif opti_config.type == 'Adam':
        # 使用optim.Adam构建优化器，传入模型参数和其他参数
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    # 如果优化器类型为'SGD'
    elif opti_config.type == 'SGD':
        # 使用optim.SGD构建优化器，传入模型参数、nesterov参数和其他参数
        optimizer = optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    # 如果优化器类型不在上述类型中
    else:
        # 抛出未实现的错误
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    else:
        raise NotImplementedError()

    # 如果配置中存在批归一化层的调度器配置
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        # 根据批归一化层调度器类型构建对应的批归一化层学习率调度器
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # 调用misc.py中的函数
        # 将学习率调度器和批归一化层学习率调度器组合成列表
        scheduler = [scheduler, bnscheduler]

    return optimizer, scheduler

# 定义一个函数，用于恢复模型
def resume_model(base_model, args, logger = None):
    # 拼接检查点路径
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    # 如果检查点路径不存在
    if not os.path.exists(ckpt_path):
        # 打印日志信息，指示没有从指定路径找到检查点文件
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        # 返回0, 0
        return 0, 0
    # 打印日志信息，指示正在从检查点路径加载模型权重
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # 加载状态字典
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    # 恢复基础模型的参数
    # 如果args.local_rank等于0
    if args.local_rank == 0:
        # 从状态字典中提取基础模型的参数
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        # 加载基础模型的状态字典
        base_model.load_state_dict(base_ckpt, strict = True)

    # 恢复参数
    # 开始的epoch为状态字典中的epoch加1
    start_epoch = state_dict['epoch'] + 1
    # 最佳指标为状态字典中的最佳指标
    best_metrics = state_dict['best_metrics']
    # 如果最佳指标不是字典类型，则转换为状态字典
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # 打印最佳指标
    # print(best_metrics)

    # 打印日志信息，指示正在恢复检查点，显示开始的epoch和最佳指标
    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    # 返回开始的epoch和最佳指标
    return start_epoch, best_metrics
def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 