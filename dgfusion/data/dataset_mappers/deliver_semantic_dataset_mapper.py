"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
"""

import copy
import logging
import os

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from oneformer.data.tokenizer import SimpleTokenizer, Tokenize
import pycocotools.mask as mask_util
import cv2

__all__ = ["DELIVERSemanticDatasetMapper"]


class DELIVERSemanticDatasetMapper:
    @configurable
    def __init__(
        self,
        modalities=["CAMERA", "LIDAR", "EVENT", "DEPTH"],
        main_modality="CAMERA",
        random_drop=[0.2, 0.2, 0.2, 0.2],
        dilation=[-1, 2, 2, -1],
        missing_mod=[None],
        target_shape=(1042, 1042, 3),
        condition_classifer=False,
        condition_text_entries=[],
        is_train=True,
        depth_on=False,
        depth_in_log_scale=True,
        *,
        name,
        num_queries,
        meta,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        task_seq_len,
        max_seq_len,
    ):
        # Initialize attributes
        self.modalities = [modality.upper() for modality in modalities]
        self.main_modality = main_modality
        assert self.modalities[0] == self.main_modality, "The first modality should be the main modality"
        assert len(self.modalities) == len(random_drop) == len(dilation), \
            "The number of modalities, random_drop, and dilation should be the same"
        
        # Rest of the initialization is unchanged
        self.random_drop = {mod: rd for mod, rd in zip(self.modalities, random_drop)}
        self.dilation = {mod: dil for mod, dil in zip(self.modalities, dilation)}
        self.missing_mod = [missing_mod.upper() for missing_mod in missing_mod]
        for missing_mod in self.missing_mod:
            assert missing_mod in self.modalities, f"Missing modality {missing_mod} not in the list of modalities"
        
        # Additional attributes for training/testing split
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.target_shape = target_shape
        self.condition_classifer = condition_classifer
        self.condition_text_entries = condition_text_entries
        self.meta = meta
        self.name = name
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.num_queries = num_queries
        
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        
        self.class_names = self.meta.stuff_classes
        self.text_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len)
        self.task_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=task_seq_len)

        self.depth_on = depth_on
        self.depth_in_log_scale = depth_in_log_scale

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if cfg.INPUT.INTERP.upper() == "NEAREST":
            interp = Image.NEAREST
        elif cfg.INPUT.INTERP.upper() == "BILINEAR":
            interp = Image.BILINEAR
        else:
            raise NotImplementedError

        if is_train:
            # Add augmentations only for training
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    interp
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
        elif cfg.DATASETS.DELIVER.CMNEXT_EQUIVALENT_EVAL:
            augs = [T.ResizeTransform(1042,1042,1024,1024, interp)]
        else:
            augs = []

        # Configuration remains largely the same
        dataset_names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST_SEMANTIC
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        missing_mod = [x.upper() for x in cfg.MODEL.TEST.MISSING_MOD if isinstance(x, str)]

        depth_on = cfg.MODEL.DEPTH_HEAD.ENABLED if is_train else cfg.MODEL.TEST.DEPTH_ON

        ret = {
            "modalities": cfg.DATASETS.DELIVER.MODALITIES,
            "main_modality": cfg.DATASETS.DELIVER.MAIN_MODALITY.upper(),
            "random_drop": cfg.DATASETS.DELIVER.RANDOM_DROP,
            "dilation": cfg.DATASETS.DELIVER.DILATION,
            "missing_mod": missing_mod,
            "condition_classifer": cfg.MODEL.CONDITION_CLASSIFIER.ENABLED,
            "condition_text_entries": cfg.MODEL.CONDITION_CLASSIFIER.CONDITION_TEXT_ENTRIES,
            "is_train": is_train,
            "depth_on": depth_on,
            "depth_in_log_scale": cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE,
            "meta": meta,
            "name": dataset_names[0],
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES - cfg.MODEL.TEXT_ENCODER.N_CTX,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def _get_texts(self, classes, num_class_obj):
        classes = list(np.array(classes))
        texts = ["an semantic photo"] * self.num_queries
        
        for class_id in classes:
            cls_name = self.class_names[class_id]
            num_class_obj[cls_name] += 1
        
        num = 0
        for i, cls_name in enumerate(self.class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        return texts

    def should_drop_modality(self, modality):
        drop_prob = self.random_drop[modality]
        return np.random.rand() < drop_prob

    def deliver_loader(self, dataset_dict):
        modality_images = {}
        for modality in self.modalities:
            if self.should_drop_modality(modality) and self.is_train:
                modality_image = np.zeros(self.target_shape, dtype=np.uint8)
            elif not self.is_train and modality in self.missing_mod:
                modality_image = np.zeros(self.target_shape, dtype=np.uint8)
            elif modality.upper() == "CAMERA":
                modality_image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            else:
                modality_image = utils.read_image(dataset_dict[f"{modality.lower()}_file_name"], format=self.img_format)
                if modality.upper() == "EVENT":
                    if not modality_image.shape[:2] == self.target_shape[:2]:
                        # eventlowres only have 512 resolution
                        modality_image = cv2.resize(modality_image, (self.target_shape[1], self.target_shape[0]),
                                                    interpolation=cv2.INTER_NEAREST)
                if "_color" in dataset_dict[f"{modality.lower()}_file_name"]:
                    raise notImplementedError(
                        "Dilation not implemented for color lidar images (would need min pooling)")
            assert modality_image.shape == self.target_shape, f"Unexpected input shape {modality_image.shape}"
            if self.dilation[modality] > 0:
                kernel = np.ones((self.dilation[modality], self.dilation[modality]), np.uint8)
                input_image = cv2.dilate(modality_image, kernel, iterations=1)
                modality_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            modality_images[modality] = modality_image
        if self.depth_on:
            gt_depth = utils.read_image(dataset_dict[f"gt_depth_file_name"], format=self.img_format)[:,:,0].copy()
            gt_depth = gt_depth.astype(np.float32)
            gt_depth = np.expand_dims(gt_depth, axis=2)
            if self.depth_in_log_scale:
                valid_idx = gt_depth > 0
                gt_depth[valid_idx] = np.log(gt_depth[valid_idx])            
            modality_images['gt_depth'] = gt_depth
        return modality_images

    def get_condition_meta_data(self, dataset_dict):
        case_map = {
            'motionblur': 'motion blur',
            'overexposure': 'overexposure',
            'underexposure': 'underexposure',
            'lidarjitter': 'lidar jitter',
            'eventlowres': 'low event camera resolution'
        }
        conditon_map = {
            'cloud': 'cloudy',
            'fog': 'foggy',
            'night': 'nighttime',
            'rain': 'rainy',
            'sun': 'sunny'
        }
        scene_info = dataset_dict["scene_info"]
        scene_info["text"] = f"A synthetic {conditon_map[scene_info['condition']]} driving scene with "
        if scene_info["case"]:
            scene_info["text"] += f"{case_map[scene_info['case']]} artifacts."
            scene_info['case'] = case_map[scene_info['case']]
        else:
            scene_info["text"] += "no artifacts."
            scene_info["case"] = ''
        return scene_info

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        modality_images = self.deliver_loader(dataset_dict)
        image = modality_images[self.main_modality]
        utils.check_image_size(dataset_dict, image)

        if self.condition_classifer:
            condition_meta_data = self.get_condition_meta_data(dataset_dict)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            if sem_seg_gt.shape[-1] == 4:
                sem_seg_gt = sem_seg_gt[..., 0]
                sem_seg_gt[sem_seg_gt != 255] = sem_seg_gt[sem_seg_gt != 255] - 1
                if sum(sum(sem_seg_gt < 0)):
                    # print(f'{dataset_dict["file_name"]} has {sum(sum(sem_seg_gt<0))} negative pixels')
                    sem_seg_gt[sem_seg_gt < 0] = 255
            else:
                raise Exception('Unexpected semantic gt format')
        else:
            sem_seg_gt = None

        is_panoptic = "pan_seg_file_name" in dataset_dict
        if is_panoptic:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None
 
        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if self.tfm_gens:
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            if sem_seg_gt is not None:
                sem_seg_gt = aug_input.sem_seg
            else:
                raise Exception('DELIVERSemanticDatasetMapper only supports semantic segmentation')

            # apply the same transformation to every modality:
            for modality in modality_images:
                if modality != self.main_modality:
                    if modality.upper() in ["CAMERA", "REF_IMAGE"]:
                        modality_images[modality] = transforms.apply_image(modality_images[modality])
                    else:
                        modality_images[modality] = transforms.apply_segmentation(modality_images[modality])
        if not self.is_train and self.tfm_gens:
            # need to set width and height acordingly for evaluating on the 1024 results:
            dataset_dict['height'] = 1024
            dataset_dict['width'] = 1024

        if self.depth_on and not self.is_train:
            dataset_dict["gt_depth"] = modality_images.pop("gt_depth").squeeze(2)

        if is_panoptic:
            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            from panopticapi.utils import rgb2id
            pan_seg_gt = rgb2id(pan_seg_gt)
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        for modality in modality_images:
            if modality != self.main_modality:
                modality_images[modality] = torch.as_tensor(
                    np.ascontiguousarray(modality_images[modality].transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0 and self.is_train:
            H, W = image.shape[-2], image.shape[-1]
            pad_h = (self.size_divisibility - H % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - W % self.size_divisibility) % self.size_divisibility
            if pad_h > 0 or pad_w > 0:
                padding_size = [0, pad_w, 0, pad_h]
                image = F.pad(image, padding_size, mode="reflect").contiguous()
                for modality in modality_images:
                    if modality != self.main_modality:
                        padding_mode = "reflect"
                        if modality.lower() in ['gt_depth']:
                            padding_mode = "constant"                            
                        modality_images[modality] = F.pad(modality_images[modality], padding_size, mode=padding_mode).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
                if is_panoptic:
                    pan_seg_gt = F.pad(
                        pan_seg_gt, padding_size, value=0
                    ).contiguous()  # 0 is the VOID panoptic label

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

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[
                classes != self.ignore_label]  # correct for deliver format. See: https://github.com/jamycheung/DELIVER/issues/11
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

        num_class_obj = {}

        for name in self.class_names:
            num_class_obj[name] = 0

        if is_panoptic:
            prob_task = np.random.uniform(0, 1.)
            if prob_task < self.semantic_prob:
                task = "The task is semantic"
                instances, text, sem_seg = self._get_semantic_dict(pan_seg_gt, image_shape, segments_info,
                                                                   num_class_obj)
            elif prob_task < self.instance_prob:
                task = "The task is instance"
                instances, text, sem_seg = self._get_instance_dict(pan_seg_gt, image_shape, segments_info,
                                                                   num_class_obj)
            else:
                task = "The task is panoptic"
                instances, text, sem_seg = self._get_panoptic_dict(pan_seg_gt, image_shape, segments_info,
                                                                   num_class_obj)
        else:
            task = "The task is semantic"
            text = self._get_texts(instances.gt_classes, num_class_obj)

        dataset_dict["sem_seg"] = torch.from_numpy(sem_seg_gt).long()
        dataset_dict["instances"] = instances
        dataset_dict["orig_shape"] = image_shape
        dataset_dict["task"] = task
        dataset_dict["text"] = text
        if is_panoptic:
            dataset_dict["thing_ids"] = self.things

        if self.condition_classifer:
            dataset_dict["condition_text"] = [condition_meta_data[key] for key in self.condition_text_entries]

        return dataset_dict


def colorize_segmentation(segmentation, palette):
    """Convert a segmentation map to a colorized RGB image."""
    h, w = segmentation.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        color_image[segmentation == label] = color

    return Image.fromarray(color_image)