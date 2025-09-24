'''
CUSTOM ENVIRONMENT CONFIG :: AKSHAY
'''

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg 
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

import yaml
from pytransform3d.rotations import quaternion_from_euler
from math import pi

# CUSTOM Robot and Asset Path Imports
from .universal_robots_ur5e_rg2 import UR5E_TEST_PD_CFG as UR5e_CFG
from .asset_filepaths import asset_filepaths as asset_paths



@configclass
class UR5eRG2CustomTableEnvCfg(DirectRLEnvCfg):
    """
    This class contains Scene, Robot, Table and Object configurations for the custom task.
    """

    # TODO ::: Modify this to correct values
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2

    # Sizes of ACTION, OBSERVATION and STATE SPACES
    action_space = 10
    observation_space = 23
    state_space = 0

    # Dict containing multiple RigidObjectCfg
    objects = {}

    # Simulation
    sim: SimulationCfg = SimulationCfg(
            dt=1 / 120,
            render_interval=decimation,
            physx=PhysxCfg(
                    solver_type=1,
                    max_position_iteration_count=192,  # Important to avoid interpenetration.
                    max_velocity_iteration_count=1,
                    bounce_threshold_velocity=0.2,
                    friction_offset_threshold=0.01,
                    friction_correlation_distance=0.00625,
                    gpu_max_rigid_contact_count=2**23,
                    gpu_max_rigid_patch_count=2**23,
                    gpu_max_num_partitions=1,  # Important for stable simulation.
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

    # --------- Robot Setup
    # Set custom UR5e with RG2 gripper as robot
    robot = UR5e_CFG.replace(prim_path=asset_paths["robot"]["prim"])
    robot.init_state = ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.88),
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
                            "finger_joint": 0.0,
                            "left_inner_finger_joint": 0.0,
                            "right_outer_knuckle_joint": 0.0,
                            "right_inner_finger_joint": 0.0,
                        },)

    # Table Setup
    table_pos = (0.0, 0.0, 0.0)
    table_rot = quaternion_from_euler((0, 0, 0),
                                      i=0, j=0, k=0,
                                      extrinsic=True   
                                     )
    table = RigidObjectCfg(
                prim_path=asset_paths["table"]["prim"],
                init_state=RigidObjectCfg.InitialStateCfg(pos=table_pos, 
                                                        rot=table_rot
                                                        ),
                spawn=UsdFileCfg(usd_path=asset_paths["table"]["path"])
            )

    # Ground Plane Setup
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Camera Setup
    height, width = 480, 640
    camera_matrix = [[612.478, 0.0, 309.723], 
                     [0.0, 612.362, 245.359], 
                     [0.0, 0.0, 1.0]]
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    focus_distance = 1.2
    pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
    # Get the camera params
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm
    # Create the Camera Config
    camera = CameraCfg(
        prim_path=asset_paths["camera"]["prim"],
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb",
                    "distance_to_image_plane",
                    "instance_id_segmentation_fast",],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, 
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), 
                                       rot=(0.5, -0.5, 0.5, -0.5), 
                                       convention="ros"),
    )

    # Objects Setup: Set up all objects in the scene
    for obj_key in asset_paths["objects"].keys():
        # Create Rigid object in scene with randomized position
        objects[obj_key] = RigidObjectCfg(
            prim_path=asset_paths["objects"][obj_key]["prim"],
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.2, 0.815), 
                                                      rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                    usd_path=asset_paths["objects"][obj_key]["path"],
                    scale=(0.1, 0.1, 0.1),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            ),
        )

    # TODO ::: Modify this to correct values
    action_scale = 7.5
    dof_velocity_scale = 0.1

    # Reward Scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0