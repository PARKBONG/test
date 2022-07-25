import math
import numpy as np
from isaacgym import gymapi, gymutil

# Color
env_color = gymapi.Vec3(247.0/255.0, 248.0/255.0, 249.0/255.0)      # white
robot_color = gymapi.Vec3(222.0/255.0, 122.0/255.0, 117.0/255.0)    # Red
obstacle_color = gymapi.Vec3(44.0/255.0, 74.0/255.0, 96.0/255.0)    # dark blue

# Initialize Gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Scout_mini Example", headless=True)

# create a simulator
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0

# physx variables
if args.physics_engine == gymapi.SIM_PHYSX:
    # sim_params.substeps = 2
    # sim_params.physx.rest_offset = 0.001
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# Create Simulate
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")
    quit()

# Create Viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Asset Description
asset_root = "../urdf"
robot_asset_file = "/ugv_gazebo_sim/scout/scout_description/urdf/scout_mini.urdf"
env_asset_file = "/base_link_description/urdf/base_link.urdf"
obstacle_asset_file = "/box_obstacle_description/urdf/box_obstacle.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False                         # if True fix the robot on the ground (default=False)
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # if remove this line visual rendering is going to down
asset_options.flip_visual_attachments = False               # if True all environment material goes vertical line (default=False)
asset_options.disable_gravity = False                       # if True don't activate gravity (default=False)
asset_options.armature = 0.01

# set up the env grid
robot_handles = []
env_handles = []
obstacle_handles = []

num_envs = 100
envs_per_row = 10   # envs_per_row = int(math.sqrt(num_envs))
spacing = 5.2
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Default Parameters
asset_options.vhacd_enabled = True
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
env_asset = gym.load_asset(sim, asset_root, env_asset_file, asset_options)
obstacle_asset = gym.load_asset(sim, asset_root, obstacle_asset_file, asset_options)

# Initialize Pose
robot_pose = gymapi.Transform()
env_pose = gymapi.Transform()
obstacle_pose = gymapi.Transform()

# Create Environments
print("Creating {} environment".format(num_envs))
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)

    # Environment Generate
    env_pose.p = gymapi.Vec3(0.69, 0.02, 0)
    env_handles.append(gym.create_actor(env, env_asset, env_pose, "bin", i, 0))
    gym.set_rigid_body_color(env, env_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, env_color)

    # Obstacle Generate
    obstacle_pose.p = gymapi.Vec3(1.5, 0.4, 0.0)
    obstacle_handles.append(gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i, 0))
    gym.set_rigid_body_color(env, obstacle_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color)

    # Actor Generate
    robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
    robot_handles.append(gym.create_actor(env, robot_asset, robot_pose, "robot", i, 0))
    gym.set_rigid_body_color(env, robot_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, robot_color)

# Camera Setting
if not args.headless:
    cam_pos = gymapi.Vec3(30.0, 0.0, 30.0)      # 멀리 보려면 z축 변경 가까이 보려면 x축 변경
    cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while True:
    # check if we should update
    t = gym.get_sim_time(sim)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update graphics transforms
    gym.step_graphics(sim)

    if not args.headless:
        # render the viewer
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

        # Check for exit condition - user closed the viewer window
        if gym.query_viewer_has_closed(viewer):
            break

print('Done')

if not args.headless:
    gym.destroy_viewer(viewer)

gym.destroy_sim(sim)