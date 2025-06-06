"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
"""

from typing import Tuple

import copy
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import ShapeSpec

from oneformer.modeling.matcher import HungarianMatcher
from oneformer.modeling.transformer_decoder.text_transformer import TextTransformer
from oneformer.modeling.transformer_decoder.oneformer_transformer_decoder import MLP
from oneformer.oneformer_model import OneFormer

from cafuser.modeling.feature_adapter.mlp import build_feature_adapter
from cafuser.modeling.condition_text_encoder.condition_text_encoder import build_condition_text_encoder
from cafuser.modeling.condition_classifier.transformer import build_condition_classifier
from cafuser.modeling.modality_fusion.prallel_cross_attention import build_modality_fusion
from cafuser.modeling.qc_to_text_projector.mlp import build_qc_to_text_projector

from dgfusion.modeling.modality_fusion.depth_token_guided_pca import DepthTokenGuidedPCA

from .modeling.criterion import SetCriterion
from .modeling.depth_head.semantic_fpn import build_depth_head
from .modeling.depth_feature_fusion.concat import build_depth_feature_fusion

@META_ARCH_REGISTRY.register()
class DGFusion(OneFormer):
    """
    DGFusion model for depth-guided multi-modal segmentation tasks.
    TODO: discription
    """

    @configurable
    def __init__(
            self,
            fusion_type: str,
            modalities: list([str]),
            main_modality: str,
            feature_adapter_enabled: bool,
            feature_adapter: nn.ModuleList,
            fusion_module: nn.Module,
            fusion_levels: list([str]),
            condition_text_encoder_module: nn.Module,
            condition_classifier_module: nn.Module,     
            qc_to_text_projector: nn.Module,
            is_analysis: bool,
            depth_head_enabled: bool,
            depth_head: nn.Module,
            depth_features: str,
            fuse_depth_features: False,
            depth_head_pre_feature_adapter: False,
            depth_as_main_modality: False,
            depth_on: bool,
            depth_adapter: None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.fusion_type = fusion_type.lower()
        self.modalities = modalities
        self.number_modalities = len(modalities)
        self.main_modality = main_modality
        self.feature_adapter_enabled = feature_adapter_enabled
        self.feature_adapter = feature_adapter if self.feature_adapter_enabled else None
        self.fusion_module = fusion_module
        self.fusion_levels = fusion_levels
        self.condition_text_encoder_module = condition_text_encoder_module
        self.condition_classifier_module = condition_classifier_module
        self.qc_to_text_projector = qc_to_text_projector
        self.is_analysis = is_analysis
        self.depth_head_enabled = depth_head_enabled
        self.depth_on = depth_on
        if self.depth_head_enabled:
            self.depth_head = depth_head
            self.depth_features = depth_features
            assert self.depth_features in ["fused", "camera", "all"], f"Invalid depth features {self.depth_features}"
            self.fuse_depth_features = fuse_depth_features
            self.depth_head_pre_feature_adapter = depth_head_pre_feature_adapter
            self.depth_as_main_modality = depth_as_main_modality
            self.depth_adapter = depth_adapter


        self.fuison_module_names = ['pca', 'querry_guided_addition']

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        main_modality = cfg.DATASETS.MODALITIES.MAIN_MODALITY.upper()

        # Check for consitancy in the config file
        if "muses" in cfg.INPUT.DATASET_MAPPER_NAME.lower():
            assert cfg.DATASETS.PIXEL_MEAN[main_modality] == cfg.MODEL.PIXEL_MEAN, f'{cfg.DATASETS.PIXEL_MEAN[main_modality]} incosistant with {cfg.MODEL.PIXEL_MEAN}'
            assert cfg.DATASETS.PIXEL_STD[main_modality] == cfg.MODEL.PIXEL_STD, f'{cfg.DATASETS.PIXEL_STD[main_modality]} incosistant with {cfg.MODEL.PIXEL_STD}'
            pixel_mean = copy.deepcopy(cfg.DATASETS.PIXEL_MEAN[main_modality])
            pixel_std = copy.deepcopy(cfg.DATASETS.PIXEL_STD[main_modality])    
        elif "deliver" in cfg.INPUT.DATASET_MAPPER_NAME.lower():
            pixel_mean = copy.deepcopy(cfg.DATASETS.DELIVER.PIXEL_MEAN[main_modality])
            pixel_std = copy.deepcopy(cfg.DATASETS.DELIVER.PIXEL_STD[main_modality])
        else:
            raise Exception
        modalities = [main_modality]
        for modality in cfg.DATASETS.MODALITIES.ORDER:
            modality_key = modality.upper()
            if modality_key != main_modality:
                if "muses" in cfg.INPUT.DATASET_MAPPER_NAME.lower():
                    if cfg.DATASETS.MODALITIES[modality_key].LOAD:
                        if modality == "REF_IMAGE":
                            modality_key = "CAMERA"
                        pixel_mean.extend(cfg.DATASETS.PIXEL_MEAN[modality_key])
                        pixel_std.extend(cfg.DATASETS.PIXEL_STD[modality_key])
                        modalities.append(modality_key)
                elif "deliver" in cfg.INPUT.DATASET_MAPPER_NAME.lower():
                    pixel_mean.extend(cfg.DATASETS.DELIVER.PIXEL_MEAN[modality_key])
                    pixel_std.extend(cfg.DATASETS.DELIVER.PIXEL_STD[modality_key])
                    modalities.append(modality_key)
                else:
                    raise Exception
        assert len(pixel_mean) / 3 == len(modalities)

        if cfg.MODEL.FEATURE_ADAPTER.ENABLED:            
            feature_adapter = build_feature_adapter(cfg, backbone.output_shape(), modalities)
        else:
            feature_adapter = None

        sem_seg_input_shape = backbone.output_shape()
        if cfg.MODEL.FUSION.TYPE.lower() == 'ConcatFusion':
            sem_seg_input_shape = {}
            for key, value in backbone.output_shape().items():
                sem_seg_input_shape[key] = ShapeSpec(channels=len(modalities) * value.channels,
                stride=value.stride)
        elif cfg.MODEL.FUSION.TYPE.lower() == 'ParallelCrossAttention':
            for key in ['NHEAD', 'WINDOW_SIZE', 'MLP_RATIO']:
                if cfg.MODEL.FEATURE_ADAPTER.NAME == "TransformerFeatureAdapter":
                    assert cfg.MODEL.FUSION.PCA[key] == cfg.MODEL.FEATURE_ADAPTER.WINDOW_TRANSFORMER.get(key), "featrue adapter and fusion should match"

        fusion_modalities = ["DEPTH"] + [m for m in modalities] if cfg.MODEL.DEPTH_HEAD.DEPTH_AS_MAIN_MODALITY else modalities
        fusion_module = build_modality_fusion(cfg, fusion_modalities, sem_seg_input_shape)
        depth_adapter = build_feature_adapter(cfg, backbone.output_shape(), ["CAMERA"]) if cfg.MODEL.DEPTH_HEAD.FEATURE_ADAPTER else None
        if depth_adapter:
            assert cfg.MODEL.DEPTH_HEAD.FEATURE_MODATLITY == "camera"
            assert cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.PRE_FEATURE_ADAPTER
        sem_seg_head = build_sem_seg_head(cfg, sem_seg_input_shape)

        if cfg.MODEL.IS_TRAIN:
            text_encoder = TextTransformer(context_length=cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH,
                                    width=cfg.MODEL.TEXT_ENCODER.WIDTH,
                                    layers=cfg.MODEL.TEXT_ENCODER.NUM_LAYERS,
                                    vocab_size=cfg.MODEL.TEXT_ENCODER.VOCAB_SIZE)
            text_projector = MLP(text_encoder.width, cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 
                                cfg.MODEL.ONE_FORMER.HIDDEN_DIM, cfg.MODEL.TEXT_ENCODER.PROJ_NUM_LAYERS)
            if cfg.MODEL.TEXT_ENCODER.N_CTX > 0:
                prompt_ctx = nn.Embedding(cfg.MODEL.TEXT_ENCODER.N_CTX, cfg.MODEL.TEXT_ENCODER.WIDTH)
            else:
                prompt_ctx = None
        else:
            text_encoder = None
            text_projector = None
            prompt_ctx = None

        task_mlp = MLP(cfg.INPUT.TASK_SEQ_LEN, cfg.MODEL.ONE_FORMER.HIDDEN_DIM,
                        cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 2)

        # Loss parameters:
        deep_supervision = cfg.MODEL.ONE_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.ONE_FORMER.NO_OBJECT_WEIGHT

        # Use depth head:
        depth_on = cfg.MODEL.TEST.DEPTH_ON

        # loss weights
        class_weight = cfg.MODEL.ONE_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.ONE_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.ONE_FORMER.MASK_WEIGHT
        contrastive_weight = cfg.MODEL.ONE_FORMER.CONTRASTIVE_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, 
                        "loss_dice": dice_weight, "loss_contrastive": contrastive_weight}

        losses = ["labels", "masks", "contrastive"]

        condition_text_encoder_module = None
        condition_classifier_module = None
        condition_temperature = None
        qc_to_text_projector = None
        if cfg.MODEL.get('CONDITION_CLASSIFIER', False):
            if cfg.MODEL.CONDITION_CLASSIFIER.ENABLED:
                in_feature_level = cfg.MODEL.CONDITION_CLASSIFIER.TRANSFORMER.IN_FEATURE
                condition_classifier_module = build_condition_classifier(cfg, backbone.output_shape(), in_feature_level)
                if cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.ENABLED:
                    condition_text_encoder_module = build_condition_text_encoder(cfg, backbone.output_shape())
                    # add condition to loss
                    losses.append("condition")
                    condition_weight = cfg.MODEL.ONE_FORMER.CONDITION_WEIGHT
                    weight_dict.update({"loss_condition": condition_weight})
                    condition_temperature = cfg.MODEL.ONE_FORMER.CONDITION_TEMPERATURE
                if cfg.MODEL.CONDITION_CLASSIFIER.TEXT_ENCODER.QC_TO_TEXT_PROJECTOR.ENABLED:
                    qc_to_text_projector = build_qc_to_text_projector(cfg, condition_classifier_module.output_shape())     

        if cfg.MODEL.DEPTH_HEAD.ENABLED:
            depth_head = build_depth_head(cfg, backbone.output_shape())
            losses.append("depth")
            weight_dict.update({"loss_depth": cfg.MODEL.DEPTH_HEAD.LOSS.WEIGHT})
        else:
            depth_head = None

        if deep_supervision and depth_on:
            dec_layers = cfg.MODEL.ONE_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            contrast_temperature=cfg.MODEL.ONE_FORMER.CONTRASTIVE_TEMPERATURE,
            condition_temperature=condition_temperature,
            losses=losses,
            num_points=cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.ONE_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO,
            depth_in_log_scale=cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE,
            si_log_ratio=cfg.MODEL.DEPTH_HEAD.LOSS.SI_LOG_RATIO,
            depth_seg_smooth_kernal=cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_SEG_SMOOTH_KERNAL,
            depth_discard_ratio=cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_DISCARD_RATIO,
            depth_ratios=cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_LOSS_RATIOS,
            depth_pixel_loss=cfg.MODEL.DEPTH_HEAD.LOSS.DEPTH_PIXEL_LOSS,
        )

        is_analysis = cfg.MODEL.IS_ANALYSIS

        if not cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.ENABLED:
            fuse_depth_features = lambda x: x # identity function
        else:
            fuse_depth_features = build_depth_feature_fusion(cfg, modalities, backbone.output_shape())

        return {
            "fusion_type": cfg.MODEL.FUSION.TYPE,
            "modalities": modalities,
            "main_modality": main_modality,
            "feature_adapter_enabled": cfg.MODEL.FEATURE_ADAPTER.ENABLED,
            "feature_adapter": feature_adapter,
            "fusion_module": fusion_module,
            "fusion_levels": cfg.MODEL.FUSION.LEVELS,
            "condition_text_encoder_module": condition_text_encoder_module,
            "condition_classifier_module": condition_classifier_module,
            "qc_to_text_projector": qc_to_text_projector,
            "depth_head_enabled": cfg.MODEL.DEPTH_HEAD.ENABLED,
            "depth_head": depth_head,
            "depth_features": cfg.MODEL.DEPTH_HEAD.FEATURE_MODATLITY,
            "depth_adapter": depth_adapter,
            "fuse_depth_features": fuse_depth_features,
            "depth_head_pre_feature_adapter": cfg.MODEL.DEPTH_HEAD.FUSE_FEATURES.PRE_FEATURE_ADAPTER,
            "depth_as_main_modality": cfg.MODEL.DEPTH_HEAD.DEPTH_AS_MAIN_MODALITY,
            "is_analysis": is_analysis,
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "task_mlp": task_mlp,
            "prompt_ctx": prompt_ctx,
            "text_encoder": text_encoder,
            "text_projector": text_projector,
            "criterion": criterion,
            "num_queries": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.TEST.PANOPTIC_ON
                or cfg.MODEL.TEST.INSTANCE_ON
            ),
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            # inference
            "semantic_on": cfg.MODEL.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.TEST.PANOPTIC_ON,
            "detection_on": cfg.MODEL.TEST.DETECTION_ON,
            "depth_on": depth_on,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "task_seq_len": cfg.INPUT.TASK_SEQ_LEN,
            "max_seq_len": cfg.INPUT.MAX_SEQ_LEN,
            "is_demo": cfg.MODEL.IS_DEMO,
        }

    def fuse_features(self, features, q_condition=None, depth_features=None):
        if self.number_modalities == 1:
            return {level: feature[self.main_modality] for level, feature in features.items()}

        if self.fusion_module.expects_q_condition and q_condition is not None:
            if self.fusion_module.expects_depth_token:  
                features = self.fusion_module(features, q_condition, depth_features)            
            else:
                features = self.fusion_module(features, q_condition)
        else:
            if self.fusion_module.expects_depth_token and depth_features is not None:  
                features = self.fusion_module(features, None, depth_features)            
            else:
                features = self.fusion_module(features)
        return features

    def split_features_by_modality(self, features):
        split_features = {}
        for level, feature in features.items():
            batch_size = feature.shape[0] // self.number_modalities
            feature = feature.view(batch_size, self.number_modalities, *feature.shape[1:])
            split_features[level] = {self.modalities[idx]: feature[:, idx, ...] for idx in range(self.number_modalities)}
        return split_features


    def forward(self, batched_inputs):
        if self.is_analysis:
            batched_inputs[0]['modalities'] = self.modalities
            for modality in batched_inputs[0]['modalities']:
                batched_inputs[0][modality] = batched_inputs[0]['image']
            batched_inputs[0]['task'] = 'The task is panoptic'
        assert len(batched_inputs[0]['modalities']) == self.number_modalities
        images = []
        for batched_input in batched_inputs:
            for i, modality in enumerate(batched_input['modalities']):
                modality_image = batched_input[modality].to(self.device).type(torch.float32)
                # self.pixel_mean and self.pixel_std: entries 0-2 are for the first modality, 3-5 for the second, etc.\
                modality_image = (modality_image - self.pixel_mean[3 * i:3 * i + 3]) / self.pixel_std[3 * i:3 * i + 3]
                images.append(modality_image)
        images = ImageList.from_tensors(images, self.size_divisibility)

        tasks = torch.cat([self.task_tokenizer(x["task"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
        tasks = self.task_mlp(tasks.float()) # this creates Q_task

        features = self.backbone(images.tensor)

        features = self.split_features_by_modality(features)

        if self.feature_adapter_enabled:
            if self.depth_head_enabled:
                if self.depth_head_pre_feature_adapter:
                    original_features ={
                        level: {key: value.clone() for key, value in subdict.items()}
                        for level, subdict in features.items()
                    }
            adapted_features = self.feature_adapter(features)
            features = adapted_features
        else:
            original_features = None

        depth_head_features = None
        if self.depth_head_enabled:
            if self.depth_head_enabled or self.fuse_depth_features:
                if self.depth_features == "camera":
                    source_features = original_features if self.depth_head_pre_feature_adapter else features
                    if self.depth_adapter:
                        source_features = {level: {"CAMERA": source_features[level]["CAMERA"]} for level in features.keys()}
                        source_features = self.depth_adapter(source_features)
                    depth_head_features = {level: source_features[level]["CAMERA"] for level in features.keys()}
                elif self.depth_features == "all":
                    depth_input = original_features if self.depth_head_pre_feature_adapter else features
                    if self.depth_head_pre_feature_adapter:
                        assert self.feature_adapter_enabled, "Feature adapter must be enabled to use depth_head_pre_feature_adapter"
                    depth_head_features = self.fuse_depth_features(depth_input)
                if self.depth_as_main_modality:
                    for level in self.fusion_levels:
                        features[level]["DEPTH"] = depth_head_features[level]

        q_condition = None
        if self.condition_classifier_module:
            q_condition = self.condition_classifier_module(features)
            if self.training:
                if self.qc_to_text_projector:
                    q_condition_contrastive_logits = self.qc_to_text_projector(q_condition)
                else:
                    q_condition_contrastive_logits = q_condition
                q_condition_contrastive_logits = {'condition_contrastive_logits': q_condition_contrastive_logits}
        features = self.fuse_features(features, q_condition, depth_head_features)

        outputs = {}

        if self.depth_head_enabled and self.depth_on:
            if self.depth_features == "fused":
                depth_head_features = features.copy()
            depth_prediction = self.depth_head(depth_head_features)
            outputs["pred_depth"] = depth_prediction

        sem_seg_outputs = self.sem_seg_head(features, tasks)        
        outputs = {**outputs, **sem_seg_outputs}

        if self.training:
            raise Exception(
                "Training code for DGFusion is not provided. Please use the inference or evaluation mode."
            )
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if self.depth_on:
                mask_pred_depth = outputs["pred_depth"]

            del outputs

            processed_results = []
            for i, data in enumerate(zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            )):
                mask_cls_result, mask_pred_result, input_per_image, image_size = data
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, input_per_image["task"])
                    processed_results[-1]["instances"] = instance_r

                if self.detection_on:
                    bbox_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, input_per_image["task"])
                    processed_results[-1]["box_instances"] = bbox_r

                if self.depth_on:
                    depth_result = mask_pred_depth
                    depth_result = F.interpolate(
                        depth_result,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    processed_results[-1]["pred_depth"] = depth_result

            return processed_results
    
    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        # confidence
        mask_pred_norm = torch.nn.functional.normalize(mask_pred, dim=0, p=1) 
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred_norm) 
        scores, labels = mask_cls.max(-1) 
        cur_prob_masks = scores.view(-1, 1, 1) * mask_pred_norm
        cur_mask_scores, cur_mask_ids = cur_prob_masks.max(0) 
        pred_class = labels[cur_mask_ids]
        indices = pred_class.unsqueeze(0)
        semseg_for_pred_class = torch.gather(semseg, 0, indices).squeeze(0)
        class_confidence = semseg_for_pred_class 
        instance_confidence = cur_mask_scores / semseg_for_pred_class 

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, class_confidence, instance_confidence, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, class_confidence, instance_confidence, segments_info