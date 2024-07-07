# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import StanfordCars
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cub import CUB


try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {config.LOCAL_RANK} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {config.LOCAL_RANK} successfully build val dataset")

    if config.DIST:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = RandomSampler(dataset_train)

    if config.DIST and (not config.TEST.SEQUENTIAL):
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )
    else:
        sampler_val = SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        dataset = datasets.ImageFolder(
            root=os.path.join(config.DATA.DATA_PATH, prefix),
            transform=build_transform(is_train, config)
        )
        nb_classes = 1000
    elif config.DATA.DATASET == 'cifar100':
        dataset = datasets.CIFAR100(
            root=config.DATA.DATA_PATH,
            train=is_train,
            transform=build_transform(is_train, config)
        )
        nb_classes = 100
    elif config.DATA.DATASET == 'stanford_cars':
        split = 'train' if is_train else 'test'
        dataset = StanfordCars(
            root=config.DATA.DATA_PATH,
            split=split,
            transform=build_transform(is_train, config)
        )
        nb_classes = 196
    elif config.DATA.DATASET == 'cub':
        split = 'train' if is_train else 'test'
        dataset = CUB(
            root=config.DATA.DATA_PATH,
            split=split,
            transform=build_transform(is_train, config)
        )
        nb_classes = 200
    else:
        raise NotImplementedError(f"Dataset {config.DATA.DATASET} not supported.")

    return dataset, nb_classes


def build_transform(is_train, config):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        return transform

    t = []
    if config.TEST.CROP:
        size = int((256 / 224) * config.DATA.IMG_SIZE)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
    else:
        t.append(
            transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                interpolation=_pil_interp(config.DATA.INTERPOLATION))
        )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
