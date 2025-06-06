"""
Author: Tim Broedermann
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/)
Adapted from: https://github.com/timbroed/cafuser
"""

import cv2
import numpy as np
import torch
from cafuser.data.dataset_mappers.muses_sdk.processing.utils import load_muses_calibration_data, load_meta_data, enlarge_points_in_image
from cafuser.data.dataset_mappers.muses_sdk.processing.lidar_processing import load_lidar_projection
from cafuser.data.dataset_mappers.muses_sdk.processing.radar_processing import load_radar_projection
from cafuser.data.dataset_mappers.muses_sdk.processing.event_camera_processing import load_event_camera_projection
from cafuser.data.dataset_mappers.muses_sdk.muses_loader import Augmentations
from matplotlib import colormaps as cm
import random
import copy
import os

class MUSES_loader:
    """
    Helper class to load the MUSES dataset.

    Args:
        modalities (dict): Dictionary with the modalities to load.
        img_format (str): Image format to use.
        load_projected (bool): Whether to load the projected images.
        calib_data (dict): Calibration data.
        relevant_meta_data (dict): Relevant metadata.
        muses_data_root (str): Root of the MUSES dataset.
    """
    def __init__(self,
                modalities_cfg=None,
                muses_data_root=None,
                is_train=True,
                img_format="RGB",
                target_shape=(1080,1920,3),
                missing_mod = [None],
                depth_in_log_scale=True,
                load_depth=False):

        self.modalities = modalities_cfg
        self.main_modality = self.modalities.MAIN_MODALITY.upper()
        modality_order = self.modalities.ORDER
        if modality_order != ["CAMERA", "LIDAR", "EVENT_CAMERA", "RADAR", "REF_IMAGE"]:
            raise NotImplementedError

        self.load_projected = {modality: self.modalities[modality].LOAD_PROJECTED for modality in self.modalities if modality in self.modalities.ORDER}
        self.muses_data_root = muses_data_root
        self.meta_data = load_meta_data(self.muses_data_root)
        self.calib_data = load_muses_calibration_data(self.muses_data_root)
        self.is_train = is_train
        self.img_format = img_format
        self.target_shape = target_shape
        if self.is_train:
            self.augmentations = Augmentations(self.modalities.AUGMENTATIONS)

        self.missing_mod = missing_mod
        assert self.modalities[self.main_modality].LOAD, f"Main modality {self.main_modality} is not loaded."

        self.depth_in_log_scale = depth_in_log_scale
        self.load_depth = load_depth

    def adapt_mod_file_name(self, modality, orig_file_name, projected=False):
        if projected:
            modality_folder = self.modalities[modality].PROJECTED.FOLDER
            extension = self.modalities[modality].PROJECTED.EXTENSION
        else:
            extension = self.modalities[modality].EXTENSION
            modality_folder = self.modalities[modality].FOLDER
        file_name = orig_file_name.replace(self.modalities.CAMERA.FOLDER, modality_folder)
        file_name = file_name.replace(self.modalities.CAMERA.EXTENSION, extension)
        return file_name

    def load_modality_from_raw(self, modality, modality_file_name, scene_meta_data, dtype=np.float32):
        if modality == "LIDAR":
            lidar_image = load_lidar_projection(modality_file_name, self.calib_data, scene_meta_data,
                                                   motion_compensation=self.modalities[modality].MOTION_COMPENSATION,
                                                   muses_root=self.muses_data_root,
                                                   enlarge_lidar_points=False)
            enlarge_lidar_points = self.modalities[modality].DILATION.ENABLED
            if enlarge_lidar_points:
                # We have negative values in the 3rd channel (height channel) and want to push all these pixels above 0 for the dilation to work as intended
                modality_image = lidar_image.copy()
                height_pixel_mask = modality_image[:,:,2] != 0.
                modality_image[height_pixel_mask, 2] += 50.

                kernel = self.modalities[modality].DILATION.KERNAL
                modality_image = enlarge_points_in_image(modality_image, kernel_shape=kernel)

                # Revert height adjustment (also the dilated pixels)
                height_pixel_mask = modality_image[:,:,2] != 0.
                modality_image[height_pixel_mask, 2] -= 50.
            else:
                modality_image = lidar_image
            return (modality_image.astype(dtype), lidar_image.astype(dtype))
        elif modality == "RADAR":
            modality_image = load_radar_projection(modality_file_name, self.calib_data, scene_meta_data,
                                                   motion_compensation=self.modalities[modality].MOTION_COMPENSATION,
                                                   muses_root=self.muses_data_root,
                                                   enlarge_radar_points=self.modalities[modality].DILATION.ENABLED,
                                                   dialtion_kernal=self.modalities[modality].DILATION.KERNAL,
                                                   intensity_threshold=self.modalities[modality].INTENSITY_THRESHOLD,
                                                   max_distance=self.modalities[modality].MAX_DISTANCE)
        elif modality == "EVENT_CAMERA":
            modality_image = load_event_camera_projection(modality_file_name, self.calib_data,
                                                   enlarge_event_camera_points=self.modalities[modality].DILATION.ENABLED,
                                                   dialtion_kernal=self.modalities[modality].DILATION.KERNAL)
        elif modality == "CAMERA" or modality == "REF_IMAGE":
            modality_image = cv2.imread(modality_file_name, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"Unknown modality {modality}.")
        return modality_image.astype(dtype)

    def load_projected_modality(self, modality, modality_file_name):
        if self.img_format == "RGB":
            if not os.path.exists(modality_file_name):
                raise FileNotFoundError(f"File {modality_file_name} does not exist.")
            modality_image = cv2.imread(modality_file_name, cv2.IMREAD_UNCHANGED)
        else:
            raise NotImplementedError(f"Image format {self.img_format} not implemented for muses.")
        if modality not in ["CAMERA", "REF_IMAGE"]:
            modality_image = modality_image.astype(np.float32)
        if self.modalities[modality].PROJECTED.SCALE_FACTOR:
            modality_image /= self.modalities[modality].PROJECTED.SCALE_FACTOR
        if self.modalities[modality].PROJECTED.SHIFT_FACTOR:
            modality_image -= self.modalities[modality].PROJECTED.SHIFT_FACTOR
        return modality_image

    def get_modality_file_name(self, dataset_dict, modality, projected=False):
        if modality == "CAMERA":
            modality_file_name = dataset_dict["file_name"]
        elif modality == "REF_IMAGE":
            if "clear/day" in dataset_dict["file_name"]:
                modality_file_name = dataset_dict["file_name"]
            else:
                modality_file_name = self.adapt_mod_file_name(modality, dataset_dict["file_name"], projected)
        else:
            modality_file_name = self.adapt_mod_file_name(modality, dataset_dict["file_name"], projected)
        return modality_file_name

    def should_drop_modality(self, modality):
        drop_prob = self.modalities[modality].RANDOM_DROP
        return np.random.rand() < drop_prob

    def get_condition_meta_data(self, dataset_dict):
        scene_name = dataset_dict['image_id']
        scene_meta_data = self.meta_data.get(scene_name)

        condition_meta_data = {}
        
        condition_mapping = {
            'rain': 'rainy',
            'fog': 'foggy',
            'snow': 'snowy',
            'clear': 'clear'
        }
        
        ground_condition_mapping = {
            'dry': 'dry',
            'wet': 'wet',
            'snow': 'snowy',
        }

        sun_level_mapping = {
            'sunlight': 'sunny',
            'overcast': 'overcast',
            'none': 'none',
            'nan': 'none'
        }
        
        # Extract and map conditions
        condition = condition_mapping.get(scene_meta_data['weather'].lower(), scene_meta_data['weather'].lower())
        ground_condition = ground_condition_mapping.get(scene_meta_data['ground_condition'], scene_meta_data['ground_condition'].lower())
        time_of_day = scene_meta_data['time_of_day'].lower()
        strength = scene_meta_data.get('precipitation_level', 'None').lower()
        precipitation = scene_meta_data.get('precipitation_tag', 'None').lower()
        sun_level = sun_level_mapping.get(str(scene_meta_data.get('sun_level', 'none')).lower())
        if sun_level == 'none':
            if time_of_day == 'night':
                sun_level = 'dark'
            else:
                tunnel = scene_meta_data.get("tunnel")
                if tunnel: 
                    sun_level = 'hidden'
                else:
                    raise Exception(f'This should not happen, need a sun level when at daytime. Scenes: {dataset_dict["image_id"]}')
                

        if strength == 'none' and precipitation == 'none':  
            precipitation_text = 'no precipitation'
        else:
            precipitation_text = f'{strength} {precipitation}'
        
        # Construct the full text description
        text = f"A {condition} driving scene at {time_of_day}time with {precipitation_text}, a {ground_condition} ground and a {sun_level} sky."

        # Populate the condition_meta_data dictionary
        condition_meta_data['condition'] = condition
        condition_meta_data['time_of_day'] = time_of_day
        condition_meta_data['strength'] = strength
        condition_meta_data['precipitation'] = precipitation
        condition_meta_data['ground_condition'] = ground_condition
        condition_meta_data['precipitation_text'] = precipitation_text
        condition_meta_data['sun_level'] = sun_level
        condition_meta_data['text'] = text
        
        return condition_meta_data
        

    def __call__(self, dataset_dict):
        if not all(self.load_projected.values()):
            scene_meta_data = self.meta_data[dataset_dict['image_id']]
        modality_images = {}
        gt_depth = None
        for modality in self.modalities.ORDER:
            if self.modalities[modality].LOAD:
                drop_modality = self.should_drop_modality(modality)
                if drop_modality and self.is_train and modality != 'LIDAR':
                    modality_image = np.zeros(self.target_shape, dtype=np.float32)
                elif not self.is_train and modality in self.missing_mod:
                    modality_image = np.zeros(self.target_shape, dtype=np.float32)
                else:
                    modality_file_name = self.get_modality_file_name(dataset_dict, modality, self.load_projected[modality])
                    if self.load_projected[modality]:
                        modality_image = self.load_projected_modality(modality, modality_file_name)                  
                        if modality == "LIDAR" and self.load_depth:
                            depth_file_name = self.get_modality_file_name(dataset_dict, "LIDAR", False)
                            gt_depth = load_lidar_projection(depth_file_name, 
                                                   self.calib_data, 
                                                   scene_meta_data,
                                                   motion_compensation=True,
                                                   muses_root=self.muses_data_root,
                                                   enlarge_lidar_points=False)
                    else:
                        modality_image = self.load_modality_from_raw(modality, modality_file_name, scene_meta_data)                    
                        if modality == "LIDAR":
                            modality_image, gt_depth = modality_image
                        else:
                            # it should be already loaded
                            pass
                    if self.modalities[modality].get("RANGE_IN_LOG_SCALE", False):
                        range_channel = modality_image[..., 0]
                        range_channel = self.appply_log_scale_to_foreground(range_channel)
                        modality_image[..., 0] = range_channel
                if modality == "LIDAR" and drop_modality and self.is_train:
                    modality_image = np.zeros(self.target_shape, dtype=np.float32)                    
                modality_images.update({modality: modality_image})
        for modality in modality_images:
            assert modality_images[modality].shape == modality_images[self.main_modality].shape, \
                (f"Loaded modality images have different shapes: {modality_images[modality].shape} vs "
                    f"{modality_images[self.main_modality].shape}.")

        # Apply augmentations
        if self.is_train:
            modality_images = self.augmentations(modality_images)

        if self.load_depth:
            if gt_depth is None:
                depth_file_name = self.get_modality_file_name(dataset_dict, "LIDAR", False)
                gt_depth = load_lidar_projection(depth_file_name, 
                                        self.calib_data, 
                                        scene_meta_data,
                                        motion_compensation=True,
                                        muses_root=self.muses_data_root,
                                        enlarge_lidar_points=False)
            gt_depth = np.expand_dims(gt_depth[..., 0], axis=-1)
            if self.depth_in_log_scale:
                gt_depth = self.appply_log_scale_to_foreground(gt_depth)
            modality_images.update({'gt_depth': gt_depth})

        return modality_images

    def appply_log_scale_to_foreground(self, channel):
        if np.any((channel > 0) & (channel < 1)):
            raise ValueError(f"Unexpected range value.")
        background_idx = channel == 0
        channel[~background_idx] = np.log(channel[~background_idx])
        return channel