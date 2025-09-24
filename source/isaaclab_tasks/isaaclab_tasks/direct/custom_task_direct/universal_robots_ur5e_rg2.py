# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Copied from isaaclab_assets/robots/universal_robots.py

Configuration for the Universal Robots UR5e with RG2 gripper.

PLACE INTO <ISAACLAB_PATH>/source/isaaclab_assets/isaaclab_assets/robots

Reference: https://github.com/ros-industrial/universal_robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .asset_filepaths import asset_filepaths as asset_paths


"""Configuration of UR5e with RG2 gripper using implicit actuator models."""
UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=asset_paths["robot"]["path"],
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.815),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Arm joints
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.0472,
            "elbow_joint": 1.0472,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
            # Gripper joints
            "left_inner_finger_joint": 0.0,
            "finger_joint": 0.0,
            #"left_inner_knuckle_finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            #"right_inner_knuckle_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "shoulder_pan_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=94.50081,
            damping=20.0, 
        ),
        "shoulder_lift_joint": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=102.91977,
            damping=20.0, 
        ),
        "elbow_joint": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=102.919, 
            damping=20.0, 
        ),
        "wrist_1_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=50.0, 
            damping=7.5, 
        ),
        "wrist_2_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=50, 
            damping=7.5, 
        ),
        "wrist_3_joint": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            effort_limit=150.0,
            velocity_limit=100.0,
            stiffness=50, 
            damping=7.5, 
        ),
        "rg2_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_finger_joint",
            ],
            effort_limit=28.0,
            velocity_limit=100.0,
            stiffness=50.0,
            damping=7.5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)



"""Configuration of Universal Robots UR5e with RG2 gripper and stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
UR5E_TEST_PD_CFG = UR5e_CFG.copy()
UR5E_TEST_PD_CFG.spawn.rigid_props.disable_gravity = False
UR5E_TEST_PD_CFG.actuators["shoulder_pan_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["shoulder_pan_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["shoulder_lift_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["shoulder_lift_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["elbow_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["elbow_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["wrist_1_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["wrist_1_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["wrist_2_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["wrist_2_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["wrist_3_joint"].stiffness *= 100
UR5E_TEST_PD_CFG.actuators["wrist_3_joint"].damping *= 40
UR5E_TEST_PD_CFG.actuators["rg2_gripper"].stiffness *= 100 
UR5E_TEST_PD_CFG.actuators["rg2_gripper"].damping *= 40