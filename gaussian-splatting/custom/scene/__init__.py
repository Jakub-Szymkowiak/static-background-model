#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import subprocess

import numpy as np

from utils.graphics_utils import BasicPointCloud

from custom.scene.dataset_readers import sceneLoadTypeCallbacks


class Scene:
    gaussians: GaussianModel

    def __init__(
            self, 
            args: ModelParams, 
            gaussians: GaussianModel, 
            load_iteration=None, 
            shuffle=True, 
            resolution_scales=[1.0],
        ):

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = sceneLoadTypeCallbacks["DAS3R+lama"](args.source_path, args.images, args.eval)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)

        if self.loaded_iter:
            path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            self.gaussians.load_ply(path)
        else:
            self.gaussians.init(cam_infos=scene_info.train_cameras, 
                                spatial_lr_scale=self.cameras_extent)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]