import os
import math
import random
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
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)    # set the normal force to be z dimension
plane_params.static_friction = 0.7
plane_params.dynamic_friction = 0.7
plane_params.restitution = 0.0
# plane_params.ground_texture = "../textures/texture_background_wall_paint_3.jpg"
gym.add_ground(sim, plane_params)

# Asset Description
asset_root = "../urdf"
robot_asset_file = "/ugv_gazebo_sim/scout/scout_description/urdf/scout_mini.urdf"
env_asset_file = "/base_link_description/urdf/base_link.urdf"
obstacle_asset_file = "/box_obstacle_description/urdf/box_obstacle.urdf"

# Load texture from file
texture_files = os.listdir("../textures/")
floor_type = "particle_board_paint_aged.jpg"
env_floor = gym.create_texture_from_file(sim, os.path.join("../textures/", floor_type))

# Environment light color
# l_color = gymapi.Vec3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1))
# l_ambient = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
# l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
# gym.set_light_parameters(sim, 0, l_color, l_ambient, l_direction)

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False                         # if True fix the robot on the ground (default=False)
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # if remove this line visual rendering is going to down, other option(FROM_ASSET)
asset_options.flip_visual_attachments = False               # if True all environment material goes vertical line (default=False)
asset_options.disable_gravity = False                       # if True don't activate gravity (default=False)
asset_options.armature = 0.01
asset_options.vhacd_enabled = True
asset_options.vhacd_params.resolution = 300000
asset_options.vhacd_params.max_convex_hulls = 10
asset_options.vhacd_params.max_num_vertices_per_ch = 64

# set up the env grid
robot_handles = []
env_handles = []
obstacle_handles = []

num_envs = 10
envs_per_row = 10   # envs_per_row = int(math.sqrt(num_envs))
spacing = 5.2
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Default Parameters
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
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "Scout_Mini", i, 0)
    robot_handles.append(robot_handle)
    # gym.set_rigid_body_color(env, robot_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, robot_color)
    gym.set_rigid_body_texture(env, robot_handles[-1], 0, gymapi.MESH_VISUAL, env_floor)

    # Set controller modes
    props = gym.get_actor_dof_properties(env, robot_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT) # other options(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, robot_handle, props)
    vel_targets = np.random.uniform(-math.pi, math.pi, 1).astype('f')
    gym.set_actor_dof_velocity_targets(env, robot_handle, vel_targets)
    efforts = np.zeros((51,), dtype=np.float32)
    # RR, RL, FR, FL = 4.0, -4.0, -4.0, 4.0
    RR, RL, FR, FL = 8.0, -8.0, -8.0, 8.0
    efforts[0] = -RR
    efforts[13] = FL
    efforts[26] = -FR
    efforts[39] = RL
    gym.apply_actor_dof_efforts(env, robot_handle, efforts)
    # props = gym.get_actor


# Camera Setting
if not args.headless:
    cam_pos = gymapi.Vec3(7.0, 0.0, 7.0)      # 멀리 보려면 z축 변경 가까이 보려면 x축 변경
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