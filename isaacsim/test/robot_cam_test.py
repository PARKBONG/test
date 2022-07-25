import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# color
tray_color = gymapi.Vec3(0.99, 0.99, 0.99)                # white
# tray_color = gymapi.Vec3(0.0, 0.11, 0.66)               # blue
# tray_color = gymapi.Vec3(0.44, 0.88, 0.66)              # green

# robot_color = gymapi.Vec3(0.0, 0.10, 0.30)              # dark blue
# robot_color = gymapi.Vec3(1.0, 127.0/255.0, 0.0)        # orange
# robot_color = gymapi.Vec3(1.0, 1.0, 0.0)                # yellow
# robot_color = gymapi.Vec3(0.0, 1.0, 0.0)                # light green
# robot_color = gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0)  # purple
# robot_color = gymapi.Vec3(139.0/255.0, 0.0, 1.0)        # pink
robot_color = gymapi.Vec3(222.0/255.0, 122.0/255.0, 117.0/255.0)    # Red

obstacle_color = gymapi.Vec3(44.0/255.0, 74.0/255.0, 96.0/255.0)     # dark blue

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Scout_mini Example", headless=True)

# create a simulator
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0

# physx variables
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 25
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.physx.rest_offset = 0.001

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)
spacing = 5.0

# set up the env grid
envs = []
robot_handles = []
camera_handles = []
tray_handles = []
obstacle_handles = []

num_envs = 100
num_object = 10
# envs_per_row = int(math.sqrt(num_envs))
envs_per_row = 10
spacing = 5.2
box_size = 0.05
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Asset description
asset_root = "../urdf"
robot_asset_file = "/ugv_gazebo_sim/scout/scout_description/urdf/scout_mini.urdf"
box_env_asset_file = "/base_link_description/urdf/base_link.urdf"
obstacle_asset_file = "/box_obstacle_description/urdf/box_obstacle.urdf"

asset_options = gymapi.AssetOptions()
# asset_options.armature = 0.001
# asset_options.fix_base_link = True
# asset_options.thickness = 0.002
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# Load materials from meshes
# asset_options.use_mesh_materials = True
# asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# Override the bogus inertia tensors and center-of-mass properties in the YCB assets.
# These flags will force the inertial properties to be recomputed from geometry.
# asset_options.override_inertia = True
# asset_options.override_com = True

# dims
# table_dims = gymapi.Vec3(0.6, 0.4, 1.0)
table_dims = gymapi.Vec3(0.6, -0.035, 1.0)

# use default parameters
asset_options.vhacd_enabled = True
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
tray_asset = gym.load_asset(sim, asset_root, box_env_asset_file, asset_options)
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
obstacle_asset = gym.load_asset(sim, asset_root, obstacle_asset_file, asset_options)

# Sensor camera properties
cam_pos = gymapi.Vec3(0.0, 3.0, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, -1.0)
cam_props = gymapi.CameraProperties()
cam_props.width = 360
cam_props.height = 360

# initial root pose for actors
robot_init_pose = gymapi.Transform()
tray_pose = gymapi.Transform()
table_pose = gymapi.Transform()
obstacle_pose = gymapi.Transform()
robot_init_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)
# tray_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 0.707107)
table_pose.p = gymapi.Vec3(0.7, 0.5 * table_dims.y + 0.001, 0.0)
obstacle_pose.p = gymapi.Vec3(1.5, 0.4, 0.5)
spawn_height = gymapi.Vec3(0.0, 0.3, 0.0)

corner = table_pose.p - table_dims * 0.5

# create envs
print("Creating {} environment".format(num_envs))
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)

    # Environment generate
    x = corner.x + table_dims.x * 0.5
    y = table_dims.y + box_size + 0.01
    z = corner.z + table_dims.z * 0.5

    tray_pose.p = gymapi.Vec3(x, y, z)
    tray_handles.append(gym.create_actor(env, tray_asset, tray_pose, "bin", i, 0))
    gym.set_rigid_body_color(env, tray_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    # for j in range(num_object):
    #     x = corner.x + table_dims.x * 0.5 + np.random.rand() * 0.35 - 0.2
    #     y = table_dims.y + box_size * 1.2 * j - 0.05
    #     z = corner.z + table_dims.z * 0.5 + np.random.rand() * 0.3 - 0.15
    #
    #     obstacle_pose.p = gymapi.Vec3(x, y, z) + spawn_height
    # Obstacle generate
    obstacle_handles.append(gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i, 0))
    gym.set_rigid_body_color(env, obstacle_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color)

    # actor generate
    # actor = gym.create_actor(env, robot_asset, robot_init_pose, 'actor', i, 0)
    robot_handles.append(gym.create_actor(env, robot_asset, robot_init_pose, "actor", i, 0))
    gym.set_rigid_body_color(env, robot_handles[-1], 0, gymapi.MESH_VISUAL_AND_COLLISION, robot_color)

# def update_robot_move(t):
#     gym.clear_lines(viewer)
#     for i in range(num_envs):

# Camera Setting
if not args.headless:
    cam_pos = gymapi.Vec3(5, 0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while not gym.query_viewer_has_closed(viewer):
    # check if we should update
    t = gym.get_sim_time(sim)
    # print("Time : {}".format(t))

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)