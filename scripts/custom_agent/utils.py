from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_from_euler_xyz
import torch
import math
from scipy.spatial.transform import Rotation as R


def get_any_frame_in_robot_base_frame(root_pose_w, any_pose_w):
    any_pos_b, any_quat_b = subtract_frame_transforms(
                                root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                                any_pose_w[:, 0:3], any_pose_w[:, 3:7]
                            )
    return any_pos_b, any_quat_b


def convert_virtual_eef_quats_to_real_eef(quats_virtual: torch.Tensor, convention="xyz"):
    """
    Converts batched quaternions from virtual EEF frame (Y+=90deg, X-=90deg W.R.T
    actual IsaacSim robot frame), to:
    - Real EEF quaternions (WXYZ)
    - Real EEF Euler angles (radians)

    Args:
        quats_virtual (torch.Tensor): (N, 4) quaternions in WXYZ format
        convention (str): Euler convention, default "xyz"

    Returns:
        eulers_real (torch.Tensor): (N, 3) Euler angles in radians
        quats_real (torch.Tensor): (N, 4) quaternions in WXYZ format
    """
    # Convert from WXYZ to XYZW for scipy
    xyzw = quats_virtual[:, [1, 2, 3, 0]].cpu().numpy()  # (N, 4)

    # Define offset rotation: virtual → real EEF frame
    offset = R.from_euler("yz", [90, -90], degrees=True)

    # Apply offset to all input quats
    rot_virtual = R.from_quat(xyzw)
    rot_real = rot_virtual * offset.inv()

    # Output 1: Euler angles in given convention
    eulers = rot_real.as_euler(convention, degrees=False)  # (N, 3)

    # Output 2: Real quaternions in XYZW → convert back to WXYZ
    quats_xyzw = rot_real.as_quat()
    quats_wxyz = torch.from_numpy(quats_xyzw[:, [3, 0, 1, 2]]).float()  # (N, 4)

    return torch.from_numpy(eulers).float(), quats_wxyz


def get_geodesic_dict():
    pass


def get_robot_joint_state():
    pass