"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
"""

import copy
import cv2 as cv2
import logging
import numpy as np
import os
import torch
from PIL import Image
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from torch.nn import functional as F

from oneformer.data.dataset_mappers.oneformer_unified_dataset_mapper import OneFormerUnifiedDatasetMapper
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
from oneformer.utils.box_ops import masks_to_boxes

from .muses_sdk.muses_loader import MUSES_loader

__all__ = ["MUSESUnifiedDatasetMapper"]

class MUSESUnifiedDatasetMapper(OneFormerUnifiedDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by CAFuser for multi-modal dense segmentation.

    This is a modified version of the OneFormerUnifiedDatasetMapper to support
    the MUSES dataset. It handles the loading of images and annotations for
    semantic and panoptic segmentation tasks. It also applies data augmentations
    and transformations to the images and annotations.
    """

    @configurable
    def __init__(self,
                 modalities_cfg=None,
                 main_modality="CAMERA",
                 muses_data_root=None,
                 condition_classifier=False,
                 condition_text_entries=[],
                 depth_on=False,
                 depth_in_log_scale=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.muses_loader = MUSES_loader(modalities_cfg, muses_data_root, kwargs['is_train'],
                                         depth_in_log_scale=depth_in_log_scale,
                                         load_depth=depth_on)
        self.main_modality = main_modality
        self.condition_classifier = condition_classifier
        self.condition_text_entries = condition_text_entries
        self.depth_on = depth_on

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if cfg.INPUT.INTERP.upper() == "NEAREST":
            interp = Image.NEAREST
        elif cfg.INPUT.INTERP.upper() == "BILINEAR":
            interp = Image.BILINEAR
        else:
            raise NotImplementedError
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                interp,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        muses_data_root = '/'.join(meta.image_root.split('/')[:-1])
        ignore_label = meta.ignore_label

        ret = {
            "modalities_cfg": cfg.DATASETS.MODALITIES,
            "main_modality": cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper(),
            "muses_data_root": muses_data_root,
            "condition_classifier": cfg.MODEL.CONDITION_CLASSIFIER.ENABLED,
            "condition_text_entries": cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES,
            "depth_on": cfg.MODEL.DEPTH_HEAD.ENABLED,
            "depth_in_log_scale": cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE,
            "is_train": is_train,
            "meta": meta,
            "name": dataset_names[0],
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "semantic_prob": cfg.INPUT.TASK_PROB.SEMANTIC,
            "instance_prob": cfg.INPUT.TASK_PROB.INSTANCE,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "OneFormerDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below   
        modality_images = self.muses_loader(dataset_dict)
        image = modality_images[self.main_modality]
        utils.check_image_size(dataset_dict, image)

        if self.condition_classifier:
            condition_meta_data = self.muses_loader.get_condition_meta_data(dataset_dict)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

        # apply the same transformation to every modality:
        for modality in modality_images:
            if modality != self.main_modality:
                if modality.upper() in ["CAMERA", "REF_IMAGE"]:
                    modality_images[modality] = transforms.apply_image(modality_images[modality])
                else:
                    modality_images[modality] = transforms.apply_segmentation(modality_images[modality])

        from panopticapi.utils import rgb2id

        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        for modality in modality_images:
            if modality != self.main_modality:
                modality_images[modality] = torch.as_tensor(np.ascontiguousarray(modality_images[modality].transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            # Get the current image size (height, width)
            image_size = (image.shape[-2], image.shape[-1])

            # Calculate padding required to make the image dimensions divisible by size_divisibility
            height_padding = (self.size_divisibility - image_size[0] % self.size_divisibility) % self.size_divisibility
            width_padding = (self.size_divisibility - image_size[1] % self.size_divisibility) % self.size_divisibility

            if height_padding > 0 or width_padding > 0:
                # Padding: (left, right, top, bottom) in PyTorch
                padding_size = [0, width_padding, 0, height_padding]

                # Pad the main image
                image = F.pad(image, padding_size, value=128).contiguous()

                # Pad each modality image except the main modality
                for modality, modality_image in modality_images.items():
                    if modality != self.main_modality:
                        modality_images[modality] = F.pad(modality_image, padding_size, value=0).contiguous()

                # Pad semantic segmentation ground truth if it exists
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

                # Pad panoptic segmentation ground truth
                if pan_seg_gt is not None:  # Ensure pan_seg_gt is checked before padding
                    pan_seg_gt = F.pad(pan_seg_gt, padding_size, value=0).contiguous()  # 0 is the VOID panoptic label


        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
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

        if "annotations" in dataset_dict:
            raise ValueError("Pemantic segmentation dataset should not have 'annotations'.")

        prob_task = np.random.uniform(0, 1.)

        num_class_obj = {}

        for name in self.class_names:
            num_class_obj[name] = 0

        if prob_task < self.semantic_prob:
            task = "The task is semantic"
            instances, text, sem_seg = self._get_semantic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        elif prob_task < self.instance_prob:
            task = "The task is instance"
            instances, text, sem_seg = self._get_instance_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)
        else:
            task = "The task is panoptic"
            instances, text, sem_seg = self._get_panoptic_dict(pan_seg_gt, image_shape, segments_info, num_class_obj)

        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg).long()
        dataset_dict["instances"] = instances
        dataset_dict["orig_shape"] = image_shape
        dataset_dict["task"] = task
        dataset_dict["text"] = text
        dataset_dict["thing_ids"] = self.things

        if self.condition_classifier:
            dataset_dict["condition_text"] = [condition_meta_data[key] for key in self.condition_text_entries]

        return dataset_dict
