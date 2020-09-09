# -*- coding=utf-8 -*-

__all__ = [
    'tiny_imagenet',
    'imagewoof2',
    'imagenette2'
]

import os
import torch
import torchvision

_default_batch_size = 32
_default_num_workers = 4


def _transform(train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ])


def tiny_imagenet(name='train',
                  batch_size=_default_batch_size,
                  num_workers=_default_num_workers):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join('datasets', 'tiny-imagenet-200', name),
        transform=_transform(name == 'train')
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             shuffle=name == 'train')
    return dataloader


def imagewoof2(name='train',
               batch_size=_default_batch_size,
               num_workers=_default_num_workers):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join('datasets', 'imagewoof2', name),
        transform=_transform(name == 'train')
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             shuffle=name == 'train')
    return dataloader


def imagenette2(name='train',
                batch_size=_default_batch_size,
                num_workers=_default_num_workers):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join('datasets', 'imagenette2', name),
        transform=_transform(name == 'train')
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             shuffle=name == 'train')
    return dataloader
