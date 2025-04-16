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


from custom.utils.vo_eval import file_interface
from custom.utils.pose_utils import quad2rotation

import torch
import torchvision


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

def getTrajectoryNorm(cam_info, scale_multiplier=1.5):
    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])  # shape (3, 1)

    cam_centers = np.hstack(cam_centers)  # shape (3, N)
    center = np.mean(cam_centers, axis=1, keepdims=True)  # shape (3, 1)

    distances = np.linalg.norm(cam_centers - center, axis=0)
    radius = np.mean(distances) * scale_multiplier

    translate = -center.flatten()

    return {"translate": translate, "radius": radius}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": np.zeros(3), "radius": 1}

def tumpose_to_c2w(tum_pose):
    T = tum_pose[:3]
    qw, qx, qy, qz = tum_pose[3:]
    quat = torch.tensor([qx, qy, qz, qw])
    R = quad2rotation(quat.unsqueeze(0)).squeeze(0).numpy()
    c2w = np.eye(4)
    c2w[:3, :3] = R 
    c2w[:3, 3] = T
    
    return c2w

def load_intrinsics_from_txt(intrinsic_path):
    intrinsics = []
    with open(intrinsic_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            K = np.array(values).reshape(3, 3)
            intrinsics.append(K)
    return intrinsics

def readCameraInfosFromPredTraj(path):
    cam_infos = []

    traj_path = os.path.join(path, "pred_traj.txt")
    traj = file_interface.read_tum_trajectory_file(traj_path)
    traj_tum = np.column_stack((traj.positions_xyz, traj.orientations_quat_wxyz))
    num_frames = traj_tum.shape[0]

    intrinsic_path = os.path.join(path, "pred_intrinsics.txt")
    intrinsics = load_intrinsics_from_txt(intrinsic_path)

    for idx in range(num_frames):
        sys.stdout.write(f"\rReading frame {idx+1}/{num_frames}")
        sys.stdout.flush()

        pose = tumpose_to_c2w(traj_tum[idx])
        R = pose[:3, :3]
        T = pose[:3, 3] 

        K = intrinsics[idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        width = int(round(cx * 2))
        height = int(round(cy * 2))

        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        image_name = f"image{idx+1}"
        image_path = os.path.join(path, "images", f"{image_name}.png")
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        depth_name = f"depth{idx+1}"
        depth_path = os.path.join(path, "depth_maps", f"{depth_name}.png")
        depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED).astype(np.float32)
        depth = cv.cvtColor(depth, cv.COLOR_RGB2GRAY) 

        cam_info = CameraInfo(uid=idx, R=R, T=T,
                              FovY=FovY, FovX=FovX,
                              image=image, depth=depth,
                              image_path=image_path,
                              image_name=image_name,
                              depth_path=depth_path,
                              width=width, height=height,
                              is_test=False, depth_params=None)

        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    
    return cam_infos

def readDAS3RSceneInfo(path, images, eval, llffhold=8):
    cam_infos_unsorted = readCameraInfosFromPredTraj(path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [
        c for idx, c in enumerate(cam_infos) 
        if idx % llffhold != 0
    ]
    
    test_cam_infos = None

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = None
    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info


sceneLoadTypeCallbacks = {
    "DAS3R+lama": readDAS3RSceneInfo
}
