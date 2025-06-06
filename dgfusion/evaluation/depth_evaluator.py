"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
"""

from typing import Optional, Union
from collections import OrderedDict
import os
import itertools
import logging
import numpy as np
import tempfile
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from detectron2.utils import comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

metric_names = ["silog", "log10", "abs_rel", "sq_rel", "rms", "log_rms", "d1", "d2", "d3"]

# -------------------------------------------------------------------
# Depth evaluation helper: computes KITTI-style errors on valid pixels
# Returns errors as a tuple:
# (silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3)
# -------------------------------------------------------------------
def compute_depth_errors(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8):
    if valid_mask.sum() == 0:
        # No valid points: return zeros (or consider raising an exception)
        return np.zeros(9, dtype=np.float32)

    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]

    # Threshold accuracies
    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    # Error metrics
    rms = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    log_rms = np.sqrt(np.mean((np.log(gt_valid) - np.log(pred_valid)) ** 2))
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    err = np.log(pred_valid) - np.log(gt_valid)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2 + 1e-8)
    log10_err = np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid)))

    # return in the order of the metric_names variable
    return np.array([silog, log10_err, abs_rel, sq_rel, rms, log_rms, d1, d2, d3], dtype=np.float32)



def _print_depth_results_table(overall_dict, bins, metric_names):
    """
    Prints the depth evaluation metrics in a terminal table.
    The table has rows corresponding to each loss metric (in the order of metric_names)
    and columns for "Overall" plus each bin (e.g. "0-30", "30-100", "100-200").
    """
    headers = ["Loss", "Overall"] + list(bins.keys())
    data = []
    for name in metric_names:
        row = [name]
        # Get the overall metric for this loss
        overall_val = overall_dict[name]
        row.append(f"{overall_val:.3f}")
        # Append the per-bin metrics for this loss
        for bin_name in bins.keys():
            key = f"{bin_name}_{name}"
            val = overall_dict.get(key, 0)
            row.append(f"{val:.3f}")
        data.append(row)

    table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
    logger.info("Depth Evaluation Results:\n" + table)


# -------------------------------------------------------------------
# DepthEvaluator class: evaluates depth predictions (torch-based) 
# with the additional capabilities:
# - Saving predicted depth maps to disk.
# - Computing evaluation metrics overall and for specified bins:
#   0-30, 30-100, 100-200.
# -------------------------------------------------------------------
class DepthEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed: bool = True,
        output_dir: Optional[str] = None,
        eps: float = 1e-6,
        depth_range: Optional[tuple] = None,
        save_depth_predictions: bool = False,
        depth_in_log_scale: bool = False,
    ):
        """
        Args:
          dataset_name: name of the dataset.
          distributed: if running distributed.
          output_dir: directory to save evaluation results and predicted depths.
          eps: minimum valid depth.
          depth_range: overall valid depth range (min_depth, max_depth).
          save_depth_predictions: if True, save predicted depth maps to disk.
        """
        self._distributed = distributed
        self._output_dir = output_dir
        self.eps = eps
        if depth_range is not None:
            self.min_depth, self.max_depth = depth_range
        else:
            self.min_depth, self.max_depth = 1, 180
        self.save_depth_predictions = save_depth_predictions
        self.depth_in_log_scale = depth_in_log_scale

        self._predictions = []  # To store per-image depth metrics
        # We'll store both overall metrics and per-bin metrics for each image.
        self.bins = {
            "0-30": (0, 30),
            "30-50": (30, 50),
            "50-100": (50, 100),
            "100-200": (100, 200)
        }

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Expects each input dict contains a key "gt_depth" for ground truth depth,
        and each output dict contains "pred_depth" for predicted depth.
        """
        # Loop over samples
        for input, output in zip(inputs, outputs):
            pred_depth = output["pred_depth"].cpu().numpy().squeeze().squeeze()
            gt_depth = input["gt_depth"] #TODO: get full scale depth here when not test (so some gt_depth with full scale? at least without processing)

            # Ensure shapes match and flatten extra dimensions if needed
            assert pred_depth.shape == gt_depth.shape, f"Shape mismatch {pred_depth.shape} vs {gt_depth.shape}"

            if self.depth_in_log_scale:
                pred_depth = np.exp(pred_depth)
                gt_depth = np.exp(gt_depth)

                # Undo the background encoding:
                background_idx = gt_depth <= 1
                gt_depth[background_idx] = 0

            # # Clamp to avoid issues with logarithms
            pred = pred_depth.clip(min=self.eps)
            gt = gt_depth.clip(min=self.eps)

            # Overall valid mask: pixels within [min_depth, max_depth] and > eps
            valid_mask = (gt > self.min_depth) & (gt < self.max_depth) & (gt > self.eps)

            # Compute overall errors if any valid pixels exist
            overall_errors = compute_depth_errors(gt, pred, valid_mask, eps=self.eps)

            # Compute errors for each bin
            bin_errors = {}
            for bin_name, (bmin, bmax) in self.bins.items():
                # For each bin, further restrict valid_mask by bin range.
                bin_mask = valid_mask & (gt >= bmin) & (gt < bmax)
                bin_errors[bin_name] = compute_depth_errors(gt, pred, bin_mask, eps=self.eps)

            # Optionally save depth prediction to disk
            if self.save_depth_predictions and self._output_dir is not None:
                file_name = os.path.basename(input["file_name"])
                file_name_png = os.path.splitext(file_name)[0] + "_depth.png"
                save_path = os.path.join(self._output_dir, "depth", file_name_png)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save log scaled depth maps so that max_depth depth is the 255 max value and min_depth is 0
                depth_map = np.log(pred)
                depth_map = (depth_map - np.log(self.min_depth)) / (np.log(self.max_depth) - np.log(self.min_depth))
                depth_map = np.clip(depth_map, 0, 1)
                depth_map = (depth_map * 255).astype(np.uint8)
                color_map = cm.plasma(depth_map)[:, :, :3]
                plt.imsave(save_path, color_map)

            # Append per-image results: overall and per-bin errors.
            self._predictions.append({
                "overall": overall_errors,
                "bins": bin_errors
            })

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            gathered = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*gathered))
            if not comm.is_main_process():
                return {}

        # Aggregate overall metrics: stack the overall arrays and compute the mean
        overall_metrics = np.mean([pred["overall"] for pred in self._predictions], axis=0)
        overall_dict = {name: overall_metrics[i] for i, name in enumerate(metric_names)}

        # Aggregate bin metrics per bin name
        for bin_name in self.bins.keys():
            # include the bin name in the metric names
            bin_metric_names = [f"{bin_name}_{name}" for name in metric_names]
            # Stack the bin arrays and compute the mean
            bin_metrics = np.mean([pred["bins"][bin_name] for pred in self._predictions], axis=0)
            overall_dict.update({name: bin_metrics[i] for i, name in enumerate(bin_metric_names)})

        # multiply abs_rel, sq_rel, d1, d2, and d3 by 100, so they are in percentage
        for key in overall_dict.keys():
            for name in ["abs_rel", "sq_rel", "d1", "d2", "d3"]:
                if name in key:
                    overall_dict[key] *= 100

        # Combine overall and bin metrics into a single dictionary
        results = OrderedDict({"depth": overall_dict})
        _print_depth_results_table(overall_dict, self.bins, metric_names)

        return results