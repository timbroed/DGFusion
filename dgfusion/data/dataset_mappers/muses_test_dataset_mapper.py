"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
"""

import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
from PIL import Image

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
from oneformer.data.dataset_mappers.dataset_mapper import DatasetMapper

from .muses_sdk.muses_loader import MUSES_loader

__all__ = ["MUSESTestDatasetMapper"]

def build_augmentation(cfg, is_train):
    """
    Adapted from utils.build_augmentation()
    """
    if cfg.INPUT.INTERP.upper() == "NEAREST":
        interp = Image.NEAREST
    elif cfg.INPUT.INTERP.upper() == "BILINEAR":
        interp = Image.BILINEAR        
    else:
        raise NotImplementedError
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style, interp)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation

class MUSESTestDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(self,
                 modalities_cfg=None,
                 main_modality="CAMERA",
                 muses_data_root=None,
                 missing_mod=[None],
                 inference_only=False,
                 depth_on=False,
                 depth_in_log_scale=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.muses_loader = MUSES_loader(modalities_cfg, muses_data_root, kwargs['is_train'],
                                         depth_in_log_scale=depth_in_log_scale,
                                         load_depth=depth_on)
        self.main_modality = main_modality
        self.depth_on = depth_on
        self.inference_only = inference_only

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        dataset_names = cfg.DATASETS.TEST_PANOPTIC
        meta = MetadataCatalog.get(dataset_names[0])
        muses_data_root = '/'.join(meta.image_root.split('/')[:-1])
        missing_mod = [x.upper() for x in cfg.MODEL.TEST.MISSING_MOD if isinstance(x, str)]

        ret = {
            "modalities_cfg": cfg.DATASETS.MODALITIES,
            "main_modality": cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper(),
            "muses_data_root": muses_data_root,
            "depth_on": cfg.MODEL.TEST.DEPTH_ON,
            "depth_in_log_scale": cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE,
            "missing_mod": missing_mod,
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "recompute_boxes": recompute_boxes,
            "task": cfg.MODEL.TEST.TASK,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        modality_images = self.muses_loader(dataset_dict)
        image = modality_images[self.main_modality]
        utils.check_image_size(dataset_dict, image)

        if self.depth_on:
            dataset_dict["gt_depth"] = modality_images.pop("gt_depth").squeeze(2)

        task = f"The task is {self.task}"
        dataset_dict["task"] = task

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if dataset_dict.get("sem_seg_file_name", False):
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        # apply the same transformation to every modality:
        for modality in modality_images:
            if modality != self.main_modality:
                if modality in ["camera", "ref_image"]:
                    modality_images[modality] = transforms.apply_image(modality_images[modality])
                else:
                    modality_images[modality] = transforms.apply_segmentation(modality_images[modality])

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor. 
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))       
        for modality in modality_images:
            if modality != self.main_modality:
                modality_images[modality] = torch.as_tensor(np.ascontiguousarray(modality_images[modality].transpose(2, 0, 1)))
        modalities = [self.main_modality]
        for modality in modality_images:
            if modality != self.main_modality:
                dataset_dict[modality] = modality_images[modality]
                if modality.lower() not in ['gt_depth']:
                    modalities.append(modality)
            else:
                dataset_dict[modality] = image
                # for backwards compatability:
                dataset_dict["image"] = image
        dataset_dict["modalities"] = modalities

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict