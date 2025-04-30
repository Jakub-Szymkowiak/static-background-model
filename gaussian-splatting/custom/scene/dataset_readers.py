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
import sys
from PIL import Image
from typing import NamedTuple, Optional

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from custom.scene.monst3r_processor import Scene, Pose, Preview
from custom.utils.vo_eval import file_interface
from custom.utils.pose_utils import quad2rotation

from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    depth: np.array
    depth_path: str
    depth_name: str
    width: int
    height: int
    is_test: Optional[str]
    depth_params: dict 


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def readMonST3RSceneInfo(path, save_preview=True):
    scene = Scene(root_path=path, conf_thrs=0.0, load=True)
    num_frames = scene.num_frames

    scene.align_poses()
    scene.create_pointcloud(downsample=1)
    scene.normalize()
    scene.trim_distant(percent=20)

    if save_preview:
        preview = Preview(scene)
        preview.render()
    
    pointcloud = scene.pointcloud

    cam_infos = []

    for frame_id in tqdm(range(num_frames), desc="Setting up cameras"):
        frame = scene.get_frame(frame_id)
        
        pose = frame.pose # c2w
        view = pose.inverse # w2c
        
        uid = frame_id + 1 

        R = view.R
        T = view.T

        width, height = round(frame.intrinsics.cx * 2), round(frame.intrinsics.cy * 2)

        FovX = focal2fov(frame.intrinsics.fx, width)
        FovY = focal2fov(frame.intrinsics.fy, height)

        image = frame.image
        depth = frame.depth
        
        image_path = frame.paths["image"]
        depth_path = frame.paths["depth"]

        image_name = os.path.basename(image_path)
        depth_name = os.path.basename(depth_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T,
                              FovY=FovY, FovX=FovX,
                              image=image,
                              image_path=image_path,
                              image_name=image_name,
                              depth=depth,
                              depth_path=depth_path,
                              depth_name=depth_name,
                              width=width, height=height,
                              is_test=False, depth_params=None)

        cam_infos.append(cam_info)
    
    test_cam_infos = None

    ptc = BasicPointCloud(points=pointcloud.xyz, 
                          colors=pointcloud.rgb, 
                          normals=pointcloud.normals)

    scene_info = SceneInfo(point_cloud=ptc, 
                           train_cameras=cam_infos, 
                           test_cameras=None, 
                           nerf_normalization=None, 
                           ply_path=None, is_nerf_synthetic=False)

    return scene_info


sceneLoadTypeCallbacks = {
    "MonST3R": readMonST3RSceneInfo
}
