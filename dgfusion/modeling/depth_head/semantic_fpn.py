"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://arxiv.org/pdf/1901.02446 semantic head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.utils.registry import Registry

DEAPTH_HEAD_REGISTRY = Registry("DEAPTH_HEAD")
DEAPTH_HEAD_REGISTRY.__doc__ = """
Registry for depth prediction head module in the CAFuser.
"""

def build_depth_head(cfg, in_channels):
    name = cfg.MODEL.DEPTH_HEAD.NAME
    return DEAPTH_HEAD_REGISTRY.get(name)(cfg, in_channels)

@DEAPTH_HEAD_REGISTRY.register()
class SemanticFPN(nn.Module):
    @configurable
    def __init__(self, cfg, in_channels, feature_strides, channels,
                 input_levels, align_corners, final_relu, device: torch.device):
        super().__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        assert len(input_levels) == len(feature_strides)

        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.channels = channels
        self.input_levels = input_levels
        self.align_corners = align_corners
        self.final_relu = final_relu
        self.device = device

        self.pre_depth_seg_stride = 4

        # Define scale heads
        self.scale_heads = nn.ModuleDict()
        for i, level in enumerate(input_levels):
            head_length = max(1, int(torch.log2(torch.tensor(feature_strides[i])) - torch.log2(torch.tensor(self.pre_depth_seg_stride))))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(
                        nn.Conv2d(self.in_channels[i] if k == 0 else self.channels, self.channels, kernel_size=3, padding=1, padding_mode='reflect'),
                        nn.GroupNorm(32, self.channels),
                        nn.ReLU(inplace=True)
                    )
                )
                if feature_strides[i] != self.pre_depth_seg_stride:
                    scale_head.append(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)
                    )
            self.scale_heads[level] = nn.Sequential(*scale_head)

        # Final depth segmentation layer
        self.depth_seg = nn.Conv2d(channels, 1, kernel_size=1)

        # Move to device
        self.to(device)

    @classmethod
    def from_config(cls, cfg, input_dims):
        device = torch.device(cfg.MODEL.DEVICE if cfg.MODEL.DEVICE else 'cuda' if torch.cuda.is_available() else 'cpu')
        in_channels = [input_dims[level].channels for level in cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS]
        in_strides = [input_dims[level].stride for level in cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS]
        return {
            'cfg': cfg,
            'in_channels': in_channels,
            'feature_strides': in_strides,
            'channels': cfg.MODEL.DEPTH_HEAD.CHANNELS,
            'input_levels': cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS,
            'align_corners': cfg.MODEL.DEPTH_HEAD.ALIGN_CORNERS,
            'final_relu': cfg.MODEL.DEPTH_HEAD.FINAL_RELU,
            'device': device,
        }

    def forward(self, inputs):
        output = self.scale_heads[self.input_levels[0]](inputs[self.input_levels[0]])
        for i in range(1, len(self.input_levels)):
            level = self.input_levels[i]
            output = output + self.scale_heads[level](inputs[level])

        output = self.depth_seg(output)

        if self.final_relu:
            output = F.relu(output)

        output = F.interpolate(
            output,
            scale_factor=4,
            mode='bilinear',
            align_corners=self.align_corners
        )

        return output
