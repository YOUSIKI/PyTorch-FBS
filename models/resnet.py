# -*- coding=utf-8 -*-

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2'
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Callable, List
from . import ChannelPruning


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  groups=groups,
                  padding=dilation,
                  dilation=dilation,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return ChannelPruning(*layers)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return ChannelPruning(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[Callable] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise ValueError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, normalization='bn', activation='relu')
        self.conv2 = conv3x3(planes, planes, normalization='bn')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[Callable] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, normalization='bn', activation='relu')
        self.conv2 = conv3x3(width, width, stride, groups, dilation, normalization='bn', activation='relu')
        self.conv3 = conv1x1(width, planes * self.expansion, normalization='bn')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or 3-element tuple')
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.conv3.norm.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Module:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, normalization='bn')
        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride=stride,
                            downsample=downsample,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet(block, layers, **kwargs):
    return ResNet(block, layers, **kwargs)


def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 128
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    kwargs['width_per_group'] = 128
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
