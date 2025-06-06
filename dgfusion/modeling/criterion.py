"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/modeling/criterion.py
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.layers.wrappers import move_device_like, shapes_to_tensor

from oneformer.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from oneformer.utils import box_ops
import torch.distributed as dist
import diffdist.functional as diff_dist
import numpy as np

def silog_loss(pred: Tensor,
               target: [Tensor],
               weight: Optional[Tensor] = None,
               eps: float = 1e-4,
               reduction: Union[str, None] = 'mean',
               avg_factor: Optional[int] = None,
               ratio: float = 0.5,
               use_depth_time_diff=False,) -> Tensor:
    # Source: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/silog_loss.py
    """Computes the Scale-Invariant Logarithmic (SI-Log) loss between
    prediction and target.
    Args:
        pred (Tensor): Predicted output.
        target (Tensor): Ground truth.
        weight (Optional[Tensor]): Optional weight to apply on the loss.
        eps (float): Epsilon value to avoid division and log(0).
        reduction (Union[str, None]): Specifies the reduction to apply to the
            output: 'mean', 'sum' or None.
        avg_factor (Optional[int]): Optional average factor for the loss.
        ratio (float): The ratio between the mean and the standard deviation
    Returns:
        Tensor: The calculated SI-Log loss.
    """
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(
        pred.clamp(min=eps))

    if torch.isnan(diff_log).any() or torch.isinf(diff_log).any():
        print('[Warning] NaN or Inf encountered in SI-Log loss')
        print(f'diff_log: {diff_log}')
        print(f'pred: {pred}')
        print(f'target: {target}')
        diff_log[torch.isnan(diff_log)] = 0.0  # Sanitize NaNs
        diff_log[torch.isinf(diff_log)] = 0.0  # Sanitize infinities

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))

    # Only consider valid elements
    valid_diff = diff_log[valid_mask]

    # Compute absolute error and its 90th percentile threshold
    abs_diff = valid_diff.abs()
    if valid_diff.numel() > 0:
        threshold = torch.quantile(abs_diff, 0.9)
    else:
        threshold = 0.0

    # Create a trim mask: only keep errors below threshold
    trim_mask = torch.zeros_like(diff_log)
    trim_mask[valid_mask] = (abs_diff <= threshold).float()

    # Zero out the errors above the threshold
    diff_log = diff_log * trim_mask.float()

    # Recompute valid mask to only include trimmed pixels
    valid_mask = valid_mask.float() * trim_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(
        dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - ratio * diff_log_mean.pow(2) + eps)

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print('[Warning] NaN or Inf encountered in SI-Log loss')
        print(f'loss: {loss}')
        print(f'diff_log_sq_mean: {diff_log_sq_mean}')
        print(f'diff_log_mean: {diff_log_mean}')
        print(f'diff_log: {diff_log}')
        print(f'pred: {pred}')
        print(f'target: {target}')
        loss[torch.isnan(loss)] = 0.0
        loss[torch.isinf(loss)] = 0.0

    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss

def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:
    # Source: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/utils.py
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def gradient_loss(pred, target):
    # Compute gradients along x and y directions
    grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
    grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
    loss_grad = torch.mean(torch.abs(grad_pred_x - grad_target_x)) + \
                torch.mean(torch.abs(grad_pred_y - grad_target_y))
    return loss_grad


def edge_aware_smoothness_loss(pred, image, ignore_value=-1):
    grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
    weight_x = torch.exp(-grad_img_x)
    weight_y = torch.exp(-grad_img_y)
    for i in range(image.shape[0]): 
        # in case the image is not loaded (the modality is dropped) we do not want any wiehgt as we do not have meaningfull gradiants
        if len(image[i].unique()) == 1:
            weight_x[i,...] = 0
            weight_y[i,...] = 0

    valid_mask = (image != ignore_value).all(dim=1, keepdim=True)
    valid_mask_x = valid_mask[:, :, :, :-1] & valid_mask[:, :, :, 1:]
    valid_mask_y = valid_mask[:, :, :-1, :] & valid_mask[:, :, 1:, :]

    # Compute the masked loss (summing only over valid pixels).
    loss_x = (grad_pred_x * weight_x * valid_mask_x.float()).sum() / (valid_mask_x.float().sum() + 1e-6)
    loss_y = (grad_pred_y * weight_y * valid_mask_y.float()).sum() / (valid_mask_y.float().sum() + 1e-6)
    loss_smooth = loss_x + loss_y
    return loss_smooth

def panoptic_edge_aware_smoothness_loss(pred, seg, ignore_value=-1, dilate_kernel_size=1):
    grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
    grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

    # TODO: right now I am having high weights in invalid regions, but I want weights of 0 there instead
    valid = (seg != ignore_value)
    valid_x = valid[:, :,  :, :-1] & valid[:, :,  :, 1:]
    valid_y = valid[:, :,  :-1, :] & valid[:, :,  1:, :]
    grad_seg_x = (seg[:, :, :, :-1] != seg[:, :, :, 1:])
    grad_seg_y = (seg[:, :, :-1, :] != seg[:, :, 1:, :])

    if dilate_kernel_size > 1:
        grad_seg_x = dilate_mask(grad_seg_x, dilate_kernel_size)
        grad_seg_y = dilate_mask(grad_seg_y, dilate_kernel_size)

    weight_x = (~grad_seg_x).int()
    weight_y = (~grad_seg_y).int()

    weight_x[~valid_x] = 0
    weight_y[~valid_y] = 0    

    loss = (grad_pred_x * weight_x).mean() + (grad_pred_y * weight_y).mean()

    return loss


def dilate_mask(mask, kernel_size=3):
    # mask: [B, 1, H, W]
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2) > 0


import matplotlib.pyplot as plt
def save_tensor_as_image(tensor, filename):
    """
    Saves a tensor of shape (B, 1, H, W) as a grayscale image.
    Only the first image in the batch is saved.
    """
    # Select the first image and squeeze the channel dimension.
    img_np = tensor[0, 0, :, :].detach().cpu().numpy()
    # Normalize to [0, 1] for visualization.
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    plt.imsave(filename, img_np, cmap='gray')

def reduce_loss(loss, reduction) -> torch.Tensor:
    # Source: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/models/losses/utils.py
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def tensors_to_batch(tensors, pad_value=0):
    # Following the padding of ImageList in detectron2
    """
    Convert a list of tensors to a batched tensor.
    The size of the batched tensor is the maximum size of the input tensors.
    The input tensors are padded with `pad_value` to match the size of the batched tensor.
    Args:
        tensors: A list of tensors.
        pad_value: Value to pad the tensors.
    Returns:
        A batched tensor.
    """
    device = tensors[0].device
    image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
    image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
    max_size = torch.stack(image_sizes_tensor).max(0).values
    if torch.all(torch.stack(image_sizes_tensor) == max_size, dim=0).all():
        batched_imgs = torch.stack(tensors, dim=0).to(device)
    else:
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
        batched_imgs = tensors[0].new_full(batch_shape, pad_value, device=device)
        for i, img in enumerate(tensors):
            batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, contrast_temperature=None,
                 condition_temperature=None, depth_in_log_scale=False, si_log_ratio=0.5, use_depth_time_diff=False, 
                 depth_seg_smooth_kernal=1, depth_discard_ratio=0, depth_ratios=[0.9,0.05,0.05], depth_pixel_loss="silog"):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.cross_entropy = nn.CrossEntropyLoss()

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrast_temperature = contrast_temperature
        if self.contrast_temperature is not None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_temperature))
            self.contrast_logit_scale = self.logit_scale
        self.condition_temperature = condition_temperature
        if self.condition_temperature is not None:
            self.condition_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / condition_temperature))
        self.depth_in_log_scale = depth_in_log_scale
        self.si_log_ratio = si_log_ratio
        self.use_depth_time_diff = use_depth_time_diff
        self.depth_seg_smooth_kernal = depth_seg_smooth_kernal
        self.depth_discard_ratio = depth_discard_ratio
        self.depth_ratios = depth_ratios
        self.depth_pixel_loss = depth_pixel_loss

    def generate_labels(self, batch_size, image_x):
        if is_dist_avail_and_initialized():
            return torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()
        else:
            return torch.arange(batch_size, dtype=torch.long, device=image_x.device)

    def normalize_features(self, features):
        return F.normalize(features.flatten(1), dim=-1)

    def compute_logits(self, image_x, text_x):
        if is_dist_avail_and_initialized():
            logits_per_img = image_x @ dist_collect(text_x).t()
            logits_per_text = text_x @ dist_collect(image_x).t()
        else:
            logits_per_img = image_x @ text_x.t()
            logits_per_text = text_x @ image_x.t()
        return logits_per_img, logits_per_text

    def compute_loss(self, logits_per_img, logits_per_text, labels, logit_scale):
        logit_scale = torch.clamp(logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)
        return loss_img + loss_text

    def loss_condition(self, outputs, targets, indices, num_masks):
        assert "condition_contrastive_logits" in outputs
        assert "condition_texts" in outputs
        
        image_x = outputs["condition_contrastive_logits"].float()
        text_x = outputs["condition_texts"]

        batch_size = image_x.shape[0]
        labels = self.generate_labels(batch_size, image_x)
        
        image_x = self.normalize_features(image_x)
        text_x = self.normalize_features(text_x)

        logits_per_img, logits_per_text = self.compute_logits(image_x, text_x)
        loss_condition = self.compute_loss(logits_per_img, logits_per_text, labels, self.condition_logit_scale)

        return {"loss_condition": loss_condition}

    def loss_contrastive(self, outputs, targets, indices, num_masks):
        assert "contrastive_logits" in outputs
        assert "texts" in outputs
        
        image_x = outputs["contrastive_logits"].float()
        text_x = outputs["texts"]

        batch_size = image_x.shape[0]
        labels = self.generate_labels(batch_size, image_x)
        
        image_x = self.normalize_features(image_x)
        text_x = self.normalize_features(text_x)

        logits_per_img, logits_per_text = self.compute_logits(image_x, text_x)
        loss_contrastive = self.compute_loss(logits_per_img, logits_per_text, labels, self.contrast_logit_scale)

        return {"loss_contrastive": loss_contrastive}

    def loss_modality(self, outputs, targets, indices, num_masks):
        assert "modality_logits" in outputs
        assert "modality_labels" in outputs

        modality_logits = outputs["modality_logits"]
        modality_labels = outputs["modality_labels"]

        loss_modality = 0
        for modality in modality_logits.keys():
            for level in modality_logits[modality].keys():
                loss_modality = loss_modality + F.cross_entropy(modality_logits[modality][level], modality_labels[modality])

        return {"loss_modality": loss_modality}


    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        ce_weight = torch.full(
            src_logits.shape[:2], self.eos_coef, dtype=torch.float32, device=src_logits.device
        )
        ce_weight[idx] = torch.tensor(1.).to(target_classes.device)

        # Deprecated: loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduce=False, reduction="none")
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")

        loss_ce = loss_ce.sum(1) / ce_weight.sum()
        loss_ce = loss_ce.sum()
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_depth(self, outputs, targets, indices, num_masks, eps=1e-6):
        assert "pred_depth" in outputs

        pred_depth = outputs["pred_depth"]
        gt_depths = [t["gt_depth"] for t in targets]
        gt_depths = tensors_to_batch(gt_depths)

        if self.depth_in_log_scale:
            log_pred = pred_depth
            log_gt = gt_depths
        else:
            log_pred = torch.log(pred_depth + eps)
            log_gt = torch.log(gt_depths + eps)

        if self.depth_pixel_loss == "silog":
            pred_depth = pred_depth.exp()
            gt_depths = gt_depths.exp()
            background_idx = gt_depths <= 1
            gt_depths[background_idx] = 0
            loss_pixel_depth = silog_loss(pred_depth, gt_depths, ratio=self.si_log_ratio, use_depth_time_diff=self.use_depth_time_diff)
        elif self.depth_pixel_loss == "l1":
            valid_mask = (gt_depths > 1).detach().float()
            errors = torch.abs(pred_depth - gt_depths) * valid_mask
            valid_errors = errors[valid_mask.bool()]
            if valid_errors.numel() > 0:
                threshold = torch.quantile(valid_errors, 1 - self.depth_discard_ratio)
            else:
                threshold = torch.tensor(0.0, device=pred_depth.device)            
            filtered_mask = ((errors <= threshold) * valid_mask)
            loss_pixel_depth = torch.sum(errors * filtered_mask) / (torch.sum(filtered_mask) + 1e-6)

        ignore_value = -1    
        images = [t["image"] for t in targets]
        images = tensors_to_batch(images, pad_value=ignore_value).float()
        loss_smooth = edge_aware_smoothness_loss(pred_depth, images, ignore_value)

        sem_segs = [t["sem_seg"].unsqueeze(0) for t in targets]
        sem_segs = tensors_to_batch(sem_segs, pad_value=255)
        batch_masks = [t["masks"].int() for t in targets]
        panoptics = []
        for i, batch_mask in enumerate(batch_masks):
            panoptic = sem_segs[i].clone()
            if len(batch_mask) != 0:
                batch_mask_idx = batch_mask.argmax(axis=0).unsqueeze(0) 
                panoptic[batch_mask_idx!=0] = batch_mask_idx[batch_mask_idx!=0] + 1000
            panoptic[panoptic == 255] = ignore_value
            panoptics.append(panoptic)
        panoptics = tensors_to_batch(panoptics, pad_value=ignore_value)

        loss_semantic_smooth = panoptic_edge_aware_smoothness_loss(pred_depth, panoptics, ignore_value, self.depth_seg_smooth_kernal)

        loss_depth = self.depth_ratios[0] * loss_pixel_depth + self.depth_ratios[1] * loss_smooth + self.depth_ratios[2] * loss_semantic_smooth

        return {"loss_depth": loss_depth}


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive,
            'condition': self.loss_condition,
            'modality': self.loss_modality,
            'depth': self.loss_depth,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ["contrastive", "condition", "modality", "depth"]:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
