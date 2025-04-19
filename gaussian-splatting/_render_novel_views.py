import torch
import torchvision
import numpy as np

import json
import os

from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.system_utils import searchForMaxIteration

from custom.scene import Scene
from custom.scene.gaussian_model import GaussianModel 
from custom.scene.dataset_readers import tumpose_to_c2w 
from custom.utils.vo_eval import file_interface


class DummyCamera:
    def __init__(self, R, T, FoVx, FoVy, W, H, fid):
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.fid = fid

class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    antialiasing = False

def parse_arguments():
    parser = ArgumentParser(description="Render novel views from turntable camera path.")
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, required=False, type=int)
    parser.add_argument("--cam_path", default=None, required=False, type=str, help="Path to camera poses")
    parser.add_argument("--perturb", action="store_true", help="Apply perturbation to provided camera poses")

    # Turntable path parameters
    parser.add_argument("--center", type=float, nargs=3, required=False, default=(0.0, 0.0, 0.0), help="Scene center coordinates")
    parser.add_argument("--radius", type=float, required=False, default=10.1, help="Distance from the orbit to the scene center")
    parser.add_argument("--offset", type=float, required=False, default=10.1, help="Height offset")
    parser.add_argument("-M", type=int, required=False, default=50, help="Number of novel views")

    # Rendering camera parameters
    parser.add_argument("--fx", type=float, required=True, help="fx")
    parser.add_argument("--fy", type=float, required=True, help="fy")
    parser.add_argument("--res", type=float, nargs=2, required=False, default=(512, 288), help="Image resolution")

    args = get_combined_args(parser)

    return model.extract(args), pipeline.extract(args), args

def generate_turntable_cam_poses(C, r, h, M):
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)

    X = C[0] + r * np.cos(theta)
    Y = np.full_like(X, h)
    Z = C[2] + r * np.sin(theta)

    positions = np.stack([X, Y, Z], axis=1) 

    extrinsics = []

    for P in positions:
        forward = C - P
        forward /= np.linalg.norm(forward)

        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)

        up = np.cross(forward, right)

        R = np.stack([right, up, -forward], axis=1)
        T = -R @ P
        pose = (R, T)

        extrinsics.append(pose)

    return extrinsics

def apply_pose_perturbation(poses, T_offset, R_offset):
    R_offset = Rotation.from_euler("xyz", R_offset, degrees=True).as_matrix()
    perturbed_poses = []

    for R, T in poses:
        R_new = R_offset @ R
        T_new = T + np.array(T_offset)
        perturbed_poses.append((R_new, T_new))

    return perturbed_poses

def load_cam_poses_from_file(path):
    traj = file_interface.read_tum_trajectory_file(path)
    xyz = traj.positions_xyz
    quat = traj.orientations_quat_wxyz
    traj_tum = np.column_stack((xyz, quat))
    M = traj_tum.shape[0]

    extrinsics = []

    for i in range(M):
        c2w = tumpose_to_c2w(traj_tum[i])
        R = c2w[:3, :3]
        T = c2w[:3, 3]

        T = -R @ T

        pose = (R, T)
        extrinsics.append(pose)

    return extrinsics

def get_cameras_from_poses(poses, fovx, fovy, w, h):
    cameras = []
    M = len(poses)
    for idx, pose in enumerate(poses):
        R, T = pose
        fid = torch.Tensor([idx / M]).cuda()
        camera = DummyCamera(R=R, T=T, 
                             FoVx=fovx, FoVy=fovy, 
                             W=w, H=h, fid=fid)
        cameras.append(camera)
    return cameras

def load_gaussians(model_args: ModelParams, iteration):
    with torch.no_grad():
        gaussians = GaussianModel(model_args.sh_degree, mode="render")

        if iteration == -1:
            load_iteration = searchForMaxIteration(os.path.join(model_args.model_path, "point_cloud"))
        else:
            load_iteration = iteration

        ply_path = os.path.join(model_args.model_path, "point_cloud", 
                                "iteration_" + str(load_iteration), 
                                "point_cloud.ply")

        gaussians.load_ply(ply_path)

        return gaussians

def save_rendering_info(mode, res, fx, fy, path, T_offset=None, R_offset=None):
    args_info = {
        "mode": mode, # if cameras or loaded or generated from a turntable path
        "resolution": res, # (width, height)
        "fx": fx,
        "fy": fy,
        "model_path": path
    }

    if mode == "loaded":
        args_info["T_offset"] = T_offset
        args_info["R_offset"] = R_offset 

    json_path = os.path.join(path, "novel_views", "args_info.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w+") as json_file:
        json.dump(args_info, json_file, indent=4)


def render_novel_views(model_path, views, gaussians, background):
    render_path = os.path.join(model_path, "novel_views")
    os.makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering novel views")):
        result = render(view, gaussians, DummyPipeline(), background)
        rendering = result["render"]

        rendering_path = os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        torchvision.utils.save_image(rendering, rendering_path)

def run_rendering_pipeline():
    model_args, pipeline_args, args = parse_arguments()

    gaussians = load_gaussians(model_args, args.iteration)
    gaussians._global_scale = torch.tensor(-13.0).float().cuda()
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)

    try:
        # use provided camera trajectory
        cam_poses = load_cam_poses_from_file(args.cam_path)
        if args.perturb:
            T_offset = [0, 0, -5]
            R_offset = [-5.0, 0, 0] # degrees
            cam_poses = apply_pose_perturbation(cam_poses, T_offset, R_offset)
            save_rendering_info(mode="loaded", T_offset=T_offset, R_offset=R_offset, 
                                res=args.res, fx=args.fx, fy=args.fy, 
                                path=model_args.model_path)
    except:
        # if no file is provided, generate turntable trajectory
        cam_poses = generate_turntable_cam_poses(args.center, args.radius, args.offset, args.M) 
        save_rendering_info(mode="turntable",
                            res=args.res, fx=args.fx, fy=args.fy, 
                            path=model_args.model_path)

    w, h = args.res
    fovx = 2 * np.arctan(w / (2 * args.fx))
    fovy = 2 * np.arctan(h / (2 * args.fy))

    cameras = get_cameras_from_poses(cam_poses, fovx, fovy, w, h)

    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_func = render_novel_views
    render_func(model_args.model_path, cameras, gaussians, background)


if __name__ == "__main__":
    run_rendering_pipeline()