"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.utils.registry import Registry

DEPTH_FEATURE_FUSION_REGISTRY = Registry("DEPTH_FEATURE_FUSION")
DEPTH_FEATURE_FUSION_REGISTRY.__doc__ = """
Registry for depth feature fusion module in the CAFuser.
"""

def build_depth_feature_fusion(cfg, modalities, in_channels):
    name = cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.NAME
    return DEPTH_FEATURE_FUSION_REGISTRY.get(name)(cfg, modalities, in_channels)


@DEPTH_FEATURE_FUSION_REGISTRY.register()
class ConcatMLPFusion(nn.Module):
    @configurable
    def __init__(self, modalities, main_modality, fusion_levels, in_channels, use_modalities_weights=False, 
                 kernel_size=1, residual=False, reduction=1, use_final_nl=False):
        super().__init__()
        self.modalities = modalities
        self.main_modality = main_modality
        self.fusion_levels = fusion_levels
        self.residual = residual
        self.use_final_nl = use_final_nl

        num_modalities = len(modalities)

        # Learnable weights per modality
        if use_modalities_weights:
            self.weights = nn.Parameter(torch.ones(num_modalities, dtype=torch.float32))
        else:
            self.weights = torch.ones(num_modalities, dtype=torch.float32)

        padding = kernel_size // 2

        # Fusion MLPs for each level using 1x1 convolutions
        self.mlps = nn.ModuleDict()
        for level in self.fusion_levels:
            channels = in_channels[level].channels
            bottleneck_dim = channels // reduction
            layers = [
                nn.Conv2d(num_modalities * channels, bottleneck_dim, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(),
                nn.Conv2d(bottleneck_dim, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(channels)
            ]
            if self.use_final_nl:
                layers.append(nn.ReLU())
            self.mlps[level] = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, cfg, modalities, in_channels):
        main_modality = cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper()
        if "USE_MODALITIES_WEIGHTS" in cfg.MODEL.FUSION.keys():
            use_modalities_weights = cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.USE_MODALITIES_WEIGHTS
        else:
            use_modalities_weights = False
        use_final_nl = cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.get("USE_FINAL_NL", False)
        return {
            'modalities': modalities,
            'main_modality': main_modality,
            'fusion_levels': cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS,
            'in_channels': in_channels,
            'use_modalities_weights': use_modalities_weights,
            'kernel_size': cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.MLP.KERNEL_SIZE,
            'residual': cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.RESIDUAL_CAMERA_FEATURES,
            'reduction': cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.MLP.REDUCTION,
            'use_final_nl': use_final_nl,
        }

    def forward(self, features):
        fused_features = {}
        for level, modality_features in features.items():
            if level in self.fusion_levels:
                M = len(self.modalities)
                BS, C, H, W = modality_features[self.main_modality].size()
                stacked_feature = torch.cat([modality_features[m] * self.weights[i] for i, m in enumerate(self.modalities)],
                                               dim=1)  # [BS, M*C, H, W]
                fused_feature = self.mlps[level](stacked_feature)  # Apply MLP

                if self.residual:
                    fused_feature = fused_feature + modality_features[self.main_modality]

                fused_features[level] = fused_feature

        return fused_features
