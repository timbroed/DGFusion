"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

from detectron2.config import CfgNode as CN

__all__ = ["add_depth_prediction_config"]


def add_depth_prediction_config(cfg):
    cfg.MODEL.TEST.DEPTH_ON = False
    cfg.MODEL.TEST.SAVE_PREDICTIONS.DEPTH = False

    cfg.MODEL.DEPTH_HEAD = CN()
    cfg.MODEL.DEPTH_HEAD.ENABLED = False
    cfg.MODEL.DEPTH_HEAD.NAME = "SemanticFPN"
    cfg.MODEL.DEPTH_HEAD.FEATURE_MODATLITY = "all" # options: fused, camera, all
    cfg.MODEL.DEPTH_HEAD.FEATURE_ADAPTER = False # use together with camera
    cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.DEPTH_HEAD.DEPTH_AS_MAIN_MODALITY = False
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES = CN()
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.ENABLED = True
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.NAME = "ConcatMLPFusion"
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.USE_MODALITIES_WEIGHTS = False
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.MLP = CN()
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.MLP.REDUCTION = 4
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.MLP.KERNEL_SIZE = 1
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.PRE_FEATURE_ADAPTER = False
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.RESIDUAL_CAMERA_FEATURES = True
    cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.USE_FINAL_NL = False
    cfg.MODEL.DEPTH_HEAD.LOSS = CN()
    cfg.MODEL.DEPTH_HEAD.LOSS.WEIGHT = 1.0
    cfg.MODEL.DEPTH_HEAD.LOSS.SI_LOG_RATIO = 0.5
    cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE = True
    cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_SEG_SMOOTH_KERNAL = 3
    cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_DISCARD_RATIO = 0.2
    cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_LOSS_RATIOS = [0.9, 0.05, 0.05] # [pixel_loss, image gradiant loss, panoptic gradiant losse]
    cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_PIXEL_LOSS = "l1"

    cfg.MODEL.DEPTH_HEAD.CHANNELS = 128 # 64 # 128, # 32
    cfg.MODEL.DEPTH_HEAD.INPUT_LEVELS = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.DEPTH_HEAD.ALIGN_CORNERS = False
    cfg.MODEL.DEPTH_HEAD.FINAL_RELU = False

    cfg.MODEL.FUSION.DT = CN()
    cfg.MODEL.FUSION.DT.FEATURE_TO_TOKEN_PROJ = True