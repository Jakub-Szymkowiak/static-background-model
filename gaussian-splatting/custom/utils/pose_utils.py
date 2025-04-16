import torch
import numpy as np
from utils.graphics_utils import fov2focal

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.) / 2.
    # sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def render_wander_path(view):
    focal_length = fov2focal(view.FoVy, view.image_height)
    R = view.R
    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]
    T = -view.T.reshape(-1, 1)
    pose = np.concatenate([R, T], -1)

    num_frames = 60
    max_disp = 5000.0  # 64 , 48

    max_trans = max_disp / focal_length  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0  # * 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)  # [np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([pose, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.Tensor(render_pose))

    return output_poses

def quad2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    # bs = quad.shape[0]
    # qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    # two_s = 2.0 / (quad * quad).sum(-1)
    # rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    # rot_mat[:, 0, 0] = 1 - two_s * (qj**2 + qk**2)
    # rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    # rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    # rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    # rot_mat[:, 1, 1] = 1 - two_s * (qi**2 + qk**2)
    # rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    # rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    # rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    # rot_mat[:, 2, 2] = 1 - two_s * (qi**2 + qj**2)
    # return rot_mat
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q).cuda()

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot