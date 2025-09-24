'''
CUSTOM AGENT :: AKSHAY

This moves to 4 different pre-set points, which are set in the 
"World" frame of IsaacLab.
'''

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms

import asyncio
import httpx
import json_numpy
from math import pi

# Encapsulated Cartesian Differential Control using IsaacLab's DiffIK
from cartesian_ctrl import CartesianDiffCtrl


VIEW_MARKERS = True


async def send_request(rgb_data, instructions):
    encoded_rgb = json_numpy.dumps(rgb_data.cpu().numpy()[0,:,:,:])
    data = {"image": encoded_rgb, 
            "instruction": instructions,
            "encoded": True}
    async with httpx.AsyncClient() as client:
        response = await client.post("http://dimelab05.eng.asu.edu:8000/act",
                                     json = data)
        #print(response.status_code, response.json())
    decoded_result = json_numpy.loads(response.json())
    
    return decoded_result


def process_for_gripper_act(ik_command, gripper_command, joint_pos_des):
    # Get the 1 DoF Gripper Action
    gripper_state = 1.0
    gripper_joint_targets = torch.zeros(6)
    # Open Gripper
    if gripper_command >= 0.75:
        # Convert 1 DoF action to 6 DoF Joint Angles for Gripper
        gripper_joint_targets = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Close Gripper
    elif gripper_command <= 0.25:
        # Convert 1 DoF action to 6 DoF Joint Angles for Gripper
        gripper_joint_targets = torch.Tensor([0.0, 47*pi/180, 40*pi/180,
                                              47*pi/180, 40*pi/180, 0.0])
    joint_pos_des[:, 6:] = gripper_joint_targets
    return joint_pos_des


def run_simulator(env: gym.Env):
    # @@@@@@ -> DEBUG -> Send camera data and command to REST server
    loop = asyncio.get_event_loop()

    sim_cfg = env.env.cfg.sim
    sim = env.env.sim


    # GOALS FOR THE ROBOT -> Size 8 (XYZ, QUAT, GRIPPER)
    ee_goals = torch.tensor([[0.35, 0.35, 1.45, 0.0, 1.0, 0.0, 0.0, 1.0],
                             [-0.35, 0.35, 1.45, 0.0, 1.0, 0.0, 0.0, 1.0],
                             [0.35, 0.5, 1.2, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [-0.35, 0.5, 1.2, 0.0, 1.0, 0.0, 0.0, 0.0],
                            ],
                            device=sim_cfg.device)


    # Get the Scene from the Gym env
    scene: InteractiveScene = env.env.scene
    # Set up the Cartesian Controllers
    cartesian_ctrl = CartesianDiffCtrl(scene, sim_cfg)

    # Visualization Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    '''world_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/World"))
    if VIEW_MARKERS == True:
        world_marker.visualize(torch.tensor([0, 0, 0]), torch.tensor([1, 0, 0, 0]))'''
    robot_base_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/base_link"))
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    current_goal_idx = 0
    # Set up IK Goal Commands
    robot = scene["robot"]
    ik_commands = torch.zeros(scene.num_envs, 
                              cartesian_ctrl.diff_ik_controller.action_dim, 
                              device=robot.device)

    # Reset Controller
    cartesian_ctrl.diff_ik_controller.reset()
    cartesian_ctrl.diff_ik_controller.set_command(ik_commands)

    # Print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Reset environment
    env.reset()
    # Get the physics time
    sim_dt = sim.get_physics_dt()

    # Set the Initial Goal
    ik_commands[:] = ee_goals[current_goal_idx, :7]
    gripper_command = ee_goals[current_goal_idx, 7]

    # ---------- SET THE GOAL IN WORLD CO-ORDINATES FOR CARTESIAN SOLVER
    cartesian_ctrl.set_goal_world(ik_commands[:, 0:3], ik_commands[:, 3:7])
    # ---------- SET THE GOAL IN WORLD CO-ORDINATES FOR CARTESIAN SOLVER

    
    # -----> SET CAMERA POSITION
    # FROM CAM INSPECTOR: quat_wxyz = [-0.6941, -0.1349, -0.1349, 0.6941] (wrt 'WORLD')
    # From the properties viewer: [-68.0, 0.0, 180.0] degrees
    # -- These properties are different, since the "WORLD" ref that camera uses-
    # -- is different to the in-scene "/World" frame
    cam_position = [0.0019, 1.79042, 1.49296]
    cam_orient = [-0.6941, -0.1349, -0.1349, 0.6941]
    scene.sensors["camera1"].set_world_poses(
            positions=torch.Tensor(cam_position).unsqueeze(0),
            orientations=torch.Tensor(cam_orient).unsqueeze(0),
            convention="world",
    )


    count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            # @@@@@@ -> DEBUG -> Send camera data and command to REST server
            '''rgb_data, inst_id_seg_data = env.env.get_camera_data()
            result = loop.run_until_complete(send_request(
                                                rgb_data, 
                                                "Pick up the yellow object.")
                                            )
            print(f"----------------------> The Predicted Angles: {result[3:6] * 180 / pi},     \
                  in radians it is {result[3:6]}")'''
            
            
            # ---------- Use Jacobian and Current Jt State to achieve Goal State (set in set_goal_world)
            joint_pos_des, ee_pos_b, ee_quat_b = cartesian_ctrl.get_joint_delta_subgoals()
            # Set NEXT JOINT POS (in the Trajectory) to achieve the DELTA value towards the goal
            joint_action = process_for_gripper_act(ik_commands, gripper_command, joint_pos_des)
            env.step(joint_action)
            
            # Get Pose and Quat Errors
            pos_err, quat_err = cartesian_ctrl.get_pose_error(ik_commands)
            

            # ---------- GOAL-REACHED CONDITION ::: Change 'ik_commands' to the next goal.
            if (count % 250 == 0) or (pos_err < 5e-3 and quat_err < 5e-2): #(pos_err < 0.1 and quat_err < 0.1):
                print(f"[DEBUG]: Goal {current_goal_idx} reached. RESETTING......")
                print(f"----------> Count: {count}, Pos Error: {pos_err}, Quat Error: {quat_err}")
                # Change Goal ID
                current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                # Run the Next Action
                ik_commands[:] = ee_goals[current_goal_idx, :7].unsqueeze(0)
                gripper_command = ee_goals[current_goal_idx, 7]
                # Reset Controller
                cartesian_ctrl.diff_ik_controller.reset()
                # change goal
                print(f"[DEBUG]: RESETTING...... Current Goal IDX is {current_goal_idx}")
                

                # ----- SET THE GOAL IN WORLD CO-ORDINATES
                cartesian_ctrl.set_goal_world(ik_commands[:, 0:3], ik_commands[:, 3:7])
                # ----- SET THE GOAL IN WORLD CO-ORDINATES


            scene.write_data_to_sim()
            # Perform Step
            sim.step()
            # Update Sim Time
            count += 1
            # Update Buffers
            scene.update(sim_dt)

            # Obtain EEF World Pose from simulation
            ee_pose_w = robot.data.body_state_w[:, 
                                                cartesian_ctrl.robot_entity_cfg.body_ids[0], 
                                                0:7]
            if VIEW_MARKERS == True:
                #robot_base_marker.visualize(ee_pos_b[:, 0:3], ee_quat_b[:, 3:7])
                # update marker positions
                ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

    # close the simulator
    env.close()



def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Run the Simulator
    run_simulator(env)
    
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()