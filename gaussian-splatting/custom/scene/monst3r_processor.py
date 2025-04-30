import json
import os

from dataclasses import dataclass
from itertools import product

import numpy as np
import open3d as o3d

from PIL import Image
from scipy.spatial.transform import Rotation
from skimage.transform import resize
from tqdm import tqdm

import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


@dataclass
class Intrinsics:
    K: np.ndarray
    
    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    def backproject(self, homogeneous: np.ndarray) -> np.ndarray:
        return (self.K_inv @ homogeneous.T).T


@dataclass 
class Pose:
    pose_vec: np.ndarray # [tx, ty, tz, | qw, qx, qy, qz]

    @property
    def q(self) -> np.ndarray:
        return self.pose_vec[3:]

    @property
    def q_xyzw(self) -> np.ndarray:
        q = self.q 
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        return q_xyzw

    @property 
    def R(self) -> np.ndarray:
        return Rotation.from_quat(self.q_xyzw).as_matrix()

    @property
    def T(self) -> np.ndarray:
        return self.pose_vec[:3]

    @property
    def homogeneous(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.T
        return mat

    @property
    def inverse(self) -> "Pose":
        R = self.R
        T = self.T
        R_inv = R.T
        T_inv = -R_inv @ T

        mat_inv = np.eye(4)
        mat_inv[:3, :3] = R_inv
        mat_inv[:3, 3] = T_inv

        return Pose.from_homogeneous(mat_inv)

    @property 
    def forward_direction(self) -> np.ndarray:
        return self.R[:, 2]

    def to_valid(self) -> "Pose":
        # CURRENT 
        M = np.array([
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  -1]
        ])

        mat = np.eye(4)
        mat[:3, :3] = M @ self.R @ M
        mat[:3, 3] = self.T

        return pose # Pose.from_homogeneous(mat)
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (self.R @ points.T).T + self.T

    @classmethod
    def from_homogeneous(cls, mat: np.ndarray) -> "Pose":
        assert mat.shape == (4, 4), f"Invalid matrix shape {mat.shape}"
        R = mat[:3, :3]
        T = mat[:3, 3]

        q = Rotation.from_matrix(R).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]])

        pose_vec = np.concatenate([T, q_wxyz])
        return cls(pose_vec)

class Frame:
    def __init__(self,
                 frame_id: int,
                 image: np.ndarray,
                 depth: np.ndarray,
                 confs: np.ndarray,
                 paths: dict,
                 intrinsics: Intrinsics,
                 pose: Pose):

        self.frame_id = frame_id
        self.image = image
        self.depth = depth
        self.confs = confs
        self.paths = paths
        self.intrinsics = intrinsics
        self.pose = pose

        self.H, self.W = self.image.shape[:2]

        assert self.image.shape[:2] == (self.H, self.W), "Image shape mismatch"
        assert self.depth.shape[:2] == (self.H, self.W), "Depth shape mismatch"
        assert self.confs.shape[:2] == (self.H, self.W), "Confs shape mismatch"

    def to_points(self, stride: int=1, conf_thrs: float=0.0):
        _downsample = lambda arr: arr[::stride, ::stride]
        image, depth, confs = _downsample(self.image), _downsample(self.depth), _downsample(self.confs)

        H, W = image.shape[:2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        us, vs = us + 0.5, vs + 0.5

        homogeneous = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
        local_dirs = self.intrinsics.backproject(homogeneous)

        points_cam = local_dirs * depth.flatten()[:, None]
        
        points_world = self.pose.transform_points(points_cam)
        
        conf_flat = confs.flatten()
        mask = conf_flat > conf_thrs

        points = points_world[mask]
        colors = image.reshape(-1, 3)[mask] / 255.0
        confidences = conf_flat[mask]

        return points, colors, confidences

@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray   
    normals: np.ndarray     

class Scene:
    def __init__(self, root_path: str, conf_thrs: float=0.6, load: bool=False):
        print(f"Processing input MonST3R scene at: {root_path}")

        self.root_path = root_path
        self.conf_thrs = conf_thrs

        self.intrinsics_path = os.path.join(root_path, "pred_intrinsics.txt")
        self.trajectory_path = os.path.join(root_path, "pred_traj.txt")

        self.intrinsics_list = self._load_intrinsics()
        self.trajectory_list = self._load_trajectory()

        self.num_frames = self._get_num_frames()

        self._frames = None
        self._pointcloud = None

        if load:
            self.load_frames()

    @property
    def pointcloud(self):
        assert self._pointcloud is not None, "Cannot access pointcloud; pointcloud hasn't been created"
        return self._pointcloud

    def get_frame(self, frame_id: int):
        frame = self._frames[frame_id]
        assert frame.frame_id == frame_id, "Unexpected frame_id mismatch"
        return frame
    
    def create_pointcloud(self, downsample: int=1):
        points, colors, normals = [], [], []
        for frame in self._frames:
            pose = frame.pose
            p, c, _ = frame.to_points(stride=downsample, conf_thrs=self.conf_thrs)
            points.append(p) 
            colors.append(c)
        
        xyz = np.concatenate(points, axis=0).astype(np.float32)
        rgb = np.concatenate(colors, axis=0).astype(np.float32)

        normals = self._estimate_normals(xyz)

        xyz = xyz

        self._pointcloud = PointCloud(xyz=xyz, rgb=rgb, normals=normals)

    def normalize(self, radius: float=0.1, mode: str="cameras"):
        assert self._pointcloud is not None, "Cannot normalize; pointcloud hasn't been created"

        if mode == "pointcloud":
            base = self._pointcloud.xyz
            center = base.mean(axis=0)
            scale = radius / np.max(np.linalg.norm(base - center, axis=1))

        elif mode == "cameras":
            base = np.array([f.pose.T for f in self._frames])
            center = base.mean(axis=0)
            scale = radius / np.max(np.linalg.norm(base - center, axis=1))

        elif mode == "hybrid":
            cam_positions = np.array([f.pose.T for f in self._frames])
            ptc = self._pointcloud.xyz

            center = cam_positions.mean(axis=0)
            scale = radius / np.max(np.linalg.norm(ptc - center, axis=1))

        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        # Normalize pointcloud
        self._pointcloud.xyz = (self._pointcloud.xyz - center) * scale

        # Normalize camera poses
        for frame in self._frames:
            pose_mat = frame.pose.homogeneous
            pose_mat[:3, 3] -= center
            pose_mat[:3, 3] *= scale
            frame.pose = Pose.from_homogeneous(pose_mat)

            frame.intrinsics.K[:2, :] *= scale

        # Normalize normals
        normals = self._pointcloud.normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        self._pointcloud.normals = normals / norms

        self._switch_opengl()

    def trim_distant(self, percent: float=10.0):
        assert self._pointcloud is not None, "Pointcloud not initialized"

        xyz = self._pointcloud.xyz
        distances = np.linalg.norm(xyz, axis=1)
        threshold = np.percentile(distances, 100 - percent)
        keep_mask = distances <= threshold

        self._pointcloud.xyz = xyz[keep_mask]
        self._pointcloud.rgb = self._pointcloud.rgb[keep_mask]
        self._pointcloud.normals = self._pointcloud.normals[keep_mask]


    def load_frames(self):
        assert self._frames is None, "Cannot load frames; frames already loaded."

        frames = []

        for idx in tqdm(range(self.num_frames), desc="Loading frames"):
            K = self.intrinsics_list[idx]
            pose_vec = self.trajectory_list[idx]

            intrinsics = Intrinsics(K)
            pose = Pose(pose_vec)
            
            paths = self._gather_paths(idx)
            image, depth, confs = self._load_data(paths)

            frame_info = Frame(frame_id=idx, image=image, depth=depth, confs=confs, paths=paths, intrinsics=intrinsics, pose=pose)

            frames.append(frame_info)

        self._frames = frames

    def align_poses(self, ref_frame_id: int=None):
        assert self._frames is not None and len(self._frames) > 0, "Cannot align poses; no frames loaded."

        if ref_frame_id is None:
            print(f"No reference frame provided for camera pose alignment; assuming middle frame alignment.")
            ref_frame_id = int(round(self.num_frames // 2))

        ref_frame = self._frames[ref_frame_id]
        assert ref_frame.frame_id == ref_frame_id, "Unexpected frame_id mismatch."

        ref_pose_mat = ref_frame.pose.homogeneous
        ref_pose_inv = np.linalg.inv(ref_pose_mat)

        for frame in self._frames:
            raw_frame_mat = frame.pose.homogeneous
            aligned_frame_mat = ref_pose_inv @ raw_frame_mat
            frame.pose = Pose.from_homogeneous(aligned_frame_mat)

    def _switch_opengl(self):
        F = np.diag([1, -1, -1])

        assert self._pointcloud is not None, "Should not switch to OpenGL coordinates before the point cloud is initiated"

        #self._pointcloud.xyz = (F @ self._pointcloud.xyz.T).T
        #self._pointcloud.normals = (F @ self._pointcloud.normals.T).T

        for frame in self._frames:
            pose = frame.pose
            R = F @ pose.R @ F
            T = F @ pose.T

            pose_mat = np.eye(4)
            pose_mat[:3, :3] = R
            pose_mat[:3, 3] = T

            frame.pose = Pose.from_homogeneous(pose_mat)

    def _estimate_normals(self, xyz: np.ndarray):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        normals = np.asarray(pcd.normals)

        return normals

    def _load_intrinsics(self):
        print(f"Loading camera intrinsics from {self.intrinsics_path}")
        intrinsics = np.loadtxt(self.intrinsics_path).reshape(-1, 3, 3).astype(np.float32)
        return [_ for _ in intrinsics]

    def _load_trajectory(self):
        print(f"Loading camera trajectory from {self.trajectory_path}")
        data = np.loadtxt(self.trajectory_path).astype(np.float32)

        assert data.shape[1] == 8, f"Expected 8 columns (index + pose); got {data.shape[1]}"

        trajectory = data[:, 1:].reshape(-1, 7)
        return [_ for _ in trajectory]

    def _get_num_frames(self):
        assert self.trajectory_list != None, "Empty trajectory"
        return len(self.trajectory_list)

    def _gather_paths(self, idx):
        image_path = os.path.join(self.root_path, "images", f"frame_{idx:04d}.png")
        depth_path = os.path.join(self.root_path, "depths", f"frame_{idx:04d}.npy")
        confs_path = os.path.join(self.root_path, "confs", f"conf_{idx}.npy")

        paths = {"image": image_path, "depth": depth_path, "confs": confs_path}
        return paths

    def _load_data(self, paths):
        image = np.array(Image.open(paths["image"]).convert("RGB"))
        depth = np.load(paths["depth"])
        confs = np.load(paths["confs"])
        return image, depth, confs
    

class Preview:
    def __init__(self, scene: Scene):
        self.scene = scene

        self.ax = None

        self.save_json = None
        self._json_data = {}

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs, ys, zs = self._verts3d
            xs2d, ys2d, zs2d = proj3d.proj_transform(xs, ys, zs, self.axes.M)
            self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
            return np.min(zs2d)

        def draw(self, renderer):
            super().draw(renderer)

    def _add_arrow(
            self,
            center: np.ndarray, 
            direction: np.ndarray,
            color: str,
            length: float=3.0, 
            alpha: float=1.0
        ):

                
        vector = direction / np.linalg.norm(direction) * length
        end = center + vector

        arrow = Preview.Arrow3D([center[0], end[0]],
                                [center[1], end[1]],
                                [center[2], end[2]],
                                mutation_scale=10,
                                lw=1.2, arrowstyle="-|>",
                                color=color, alpha=0.9)

        self.ax.add_artist(arrow)

    def _add_coordinate_frame(
            self,
            origin: np.ndarray, 
            dirs: np.ndarray, 
            length=1.5, alpha=1.0
        ):

        colors = ["r", "g", "b"]
        for i in range(3):
            axis = dirs[:, i] * length
            arrow = Preview.Arrow3D([origin[0], origin[0] + axis[0]],
                            [origin[1], origin[1] + axis[1]],
                            [origin[2], origin[2] + axis[2]],
                            mutation_scale=15, 
                            lw=2, arrowstyle="-|>", 
                            color=colors[i], alpha=alpha)

            self.ax.add_artist(arrow)

    def _add_pointcloud(self, ptc: PointCloud, alpha=0.01):
        pts, rgb = ptc.xyz, ptc.rgb
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                        c=rgb, s=0.2, alpha=alpha)

    def _add_pose_coordinate_frame(self, pose: Pose):
        self._add_coordinate_frame(pose.T, pose.R)

        if self.save_json:
            if "poses" not in self._json_data:
                self._json_data["poses"] = []
                
            self._json_data["poses"].append({
                "position": pose.T.tolist(),
                "rotation": pose.R.tolist()
            })

    def _add_bounding_box(
            self, 
            ptc: PointCloud, 
            color="k", 
            lw=0.7, 
            alpha=0.5
        ):

        xyz = ptc.xyz
        min_pt, max_pt = np.min(xyz, axis=0), np.max(xyz, axis=0)

        corners = np.array(list(product(*zip(min_pt, max_pt))))
        edges = [(i, j) for i in range(8) for j in range(i+1, 8) if bin(i ^ j).count("1") == 1]
        
        for i, j in edges:
            self.ax.plot(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha)

        if self.save_json:
            self._json_data["bounding_box"] = {
                "min": min_pt.tolist(),
                "max": max_pt.tolist()
            }

    def _add_camera_trace(
            self, 
            frames: list, 
            downsample: int = 1, 
            color="red", size=10, alpha=0.8
        ):

        centers = np.array([f.pose.T for f in frames[::downsample]])
        self.ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                        c=color, s=size, marker="o", alpha=alpha)

    def _construct_geometry(
            self,
            display_bounding_box: bool=False,
            display_camera_trace: bool=False,
            display_coordinate_system: bool=False,
            display_frame_poses: bool=False,
            display_pointcloud: bool=True,
            frames_downsample: int=1,
            save_json: bool=False
        ):

        self.save_json = save_json

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("white")

        if display_coordinate_system:
            origin = np.zeros(3)
            directions = np.eye(3) 
            self._add_coordinate_frame(origin, directions)
            
        if display_pointcloud:
            self._add_pointcloud(self.scene.pointcloud)

        for frame in self.scene._frames[::frames_downsample]:
            pose = frame.pose
            if display_frame_poses:
                self._add_pose_coordinate_frame(pose)

        if display_bounding_box:
            self._add_bounding_box(self.scene.pointcloud)

        if display_camera_trace:
            self._add_camera_trace(self.scene._frames, downsample=frames_downsample)



    def _save_figure(
            self, 
            path: str,
            figsize: tuple=(16,16),
            axes_labels: dict=None,
            axes_range: tuple=(-3,3),
            elev: int=20, 
            azim: int=60,
            dist: float=10,
            dpi: int=300,
            fontsize: int=8,
            ticksize: int=6
        ):

        print(f"Saving scene preview at {path}")

        # Axes labels
        if axes_labels is None:
            axes_labels = {"X": "X", "Y": "Y", "Z": "Z"}

        self.ax.set_xlabel(axes_labels["X"], fontsize=fontsize)
        self.ax.set_ylabel(axes_labels["Y"], fontsize=fontsize)
        self.ax.set_zlabel(axes_labels["Z"], fontsize=fontsize)

        # Tick size
        self.ax.tick_params(axis="both", which="major", labelsize=ticksize)

        # Axes range
        self.ax.set_xlim(axes_range)
        self.ax.set_ylim(axes_range)
        self.ax.set_zlim(axes_range)

        # View perspective
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.dist = dist


        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()

        if self.save_json:
            json_path = path.replace(".png", ".json")
            with open(json_path, "w") as f:
                json.dump(self._json_data, f, indent=4)

        print(f"Preview saved.")

    def render(self):
        self._construct_geometry(display_bounding_box=True,
                                 display_camera_trace=True,
                                 display_coordinate_system=True,
                                 display_frame_poses=False,
                                 display_pointcloud=True,
                                 frames_downsample=6,
                                 save_json=True)

        self._save_figure("scene_preview.png", dist=10.0)
