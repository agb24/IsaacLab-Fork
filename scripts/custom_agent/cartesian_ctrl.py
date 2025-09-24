from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import torch

from utils import get_any_frame_in_robot_base_frame


class CartesianDiffCtrl():
    """
    Class encapsulates necessary functions for computing the necessary parameters 
    for IsaacLab's Differential Inverse Kinematics controller and 
    computing EEF positions.
    """
    def __init__(self,
                 scene, 
                 sim_cfg,
                 robot_entity_cfg=SceneEntityCfg("robot", 
                                                 joint_names=[".*"], 
                                                 body_names=["wrist_3_link"]),
                ):
        self.robot = scene["robot"]
        self.robot_entity_cfg = robot_entity_cfg
        self.robot_entity_cfg.resolve(scene)
        
        # Initialize the IK Controller
        self._setup_controller(scene, sim_cfg) 

    def _setup_controller(self, scene, sim_cfg):
        # Set up the Inverse Kinematic solver
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose",
                                                  use_relative_mode=False,
                                                  ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, 
                                                           num_envs=scene.num_envs, 
                                                           device=sim_cfg.device)

    def set_goal_world(self, goal_pos_w, goal_quat_w):
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
                                        root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                                        goal_pos_w, goal_quat_w
                                    )
        goal_pose_b = torch.cat([goal_pos_b, goal_quat_b], dim=-1)
        self.diff_ik_controller.set_command(goal_pose_b)

    def get_robot_state(self):
        """
        RETURNS: Robot Root State in World frame,
                 EEF in World frame
                 EEF in Robot root frame
        """
        # Robot Root State in World frame: [pos, quat, lin_vel, ang_vel] -> Shape (num_inst, 13)
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        # EEF in World frame: [pos, quat, lin_vel, ang_vel] -> Shape (num_inst, num_bodies, 13)
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        # Compute EEF in Base (Robot root) frame
        ee_pos_b, ee_quat_b = get_any_frame_in_robot_base_frame(root_pose_w, ee_pose_w)

        return root_pose_w, ee_pose_w, ee_pos_b, ee_quat_b

    def get_joint_delta_subgoals(self):
        """
        1.Get Jacobian  
        2.Get Curr Jt State  
        3.Get EEF pos in base frame  
        4.Compute & Return Jt Delta Cmds towards reaching overall goal (created in )
        """


        # ----------- Get robot parameters
        # Obtain the frame index of the EEF -> For fixed-base, frame index is (body_index -1)
        if self.robot.is_fixed_base:
            ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # ----------- GET Robot State
        root_pose_w, ee_pose_w, ee_pos_b, ee_quat_b = self.get_robot_state()
        
        # ----------- Compute values required for IK 
        # Jacobian shape: (articulation_index, jacobian_shape.numCols, jacobian_shape.numRows)
        jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, 
                                                            self.robot_entity_cfg.joint_ids]
        # Joint State of the robot -> Shape (num_instances, num_joints)
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # Compute the joint commands -> Shape (num_instances, num_joints)
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, 
                                                        jacobian, joint_pos)
        return joint_pos_des, ee_pos_b, ee_quat_b

    def get_pose_error(self, ik_commands):
        # Get the goal positions
        goal_pos, goal_quat = ik_commands[:, 0:3], ik_commands[:, 3:7]
        # ----------- GET Robot State
        root_pose_w, ee_pose_w, ee_pos_b, ee_quat_b = self.get_robot_state()
        # Get goal positions in Robot Co-ord
        robot_root_pose_w = self.robot.data.root_state_w[:, 0:7]
        goal_pos_b, goal_quat_b = get_any_frame_in_robot_base_frame(
                                                    robot_root_pose_w, 
                                                    torch.cat((goal_pos, goal_quat), 
                                                                dim=1),
                                                    )
        # Get Position Error
        pos_err = torch.norm(ee_pos_b - goal_pos_b)
        quat_err = 1.0 - torch.abs(torch.sum(ee_quat_b * goal_quat_b, dim=-1))

        return pos_err, quat_err


class CartesianGlobalCtrl():
    """
    This is a class written for Inverse Kinematics using Global Optimization.
    The DH Parameters correspond to the parameters for UR5e Robot, 
    obtained from the Universal Robots website.
    """
    def __init__(self):
        self.dh_params = [
            (0,  0.1625,  0,     np.pi/2),  
            (0,  0,      -0.425,  0),       
            (0,  0,      -0.3922, 0),       
            (0,  0.1333,  0,     np.pi/2),  
            (0,  0.0997,  0,    -np.pi/2),  
            (0,  0.0996,  0,     0)
        ]

    def rpy_to_matrix(self, rpy):
        return R.from_euler('xyz', rpy, degrees=True).as_matrix()
    
    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d],
            [0,   0,        0,       1]
        ])
    
    def forward_kinematics(self, dh_params, joint_angles, base_transform=None):
        if base_transform is None:
            T = np.eye(4)
        else:
            T = base_transform.copy()

        for i, (theta, d, a, alpha) in enumerate(dh_params):
            T_i = self.dh_transform(joint_angles[i] + theta, d, a, alpha)
            T = np.dot(T, T_i)
        return T
    
    def ik_objective(self, q, target_pose, base_transform=None):
        T_fk = self.forward_kinematics(self.dh_params, q, base_transform)
        pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
        rot_error = np.linalg.norm(T_fk[:3, :3] - target_pose[:3, :3])
        return 1.0 * pos_error + 0.1 * rot_error

    def compute_ik(self, position, rpy, 
                   q_guess=None, base_transform=None, 
                   max_tries=5, dx=0.001):
        if q_guess is None:
            q_guess = np.radians([85, -80, 90, -90, -90, 90])

        original_position = np.array(position)

        for i in range(max_tries):
            perturbed_position = original_position.copy()
            perturbed_position[0] += i * dx  # perturb x axis

            target_pose = np.eye(4)
            target_pose[:3, 3] = perturbed_position
            target_pose[:3, :3] = self.rpy_to_matrix(rpy)

            joint_bounds = [(-np.pi, np.pi)] * 6

            result = minimize(
                self.ik_objective,
                q_guess,
                args=(target_pose, base_transform),
                method='L-BFGS-B',
                bounds=joint_bounds
            )

            if result.success:
                return result.x

        print(f"IK failed after {max_tries} attempts. Tried perturbing from {original_position}.")
        return None