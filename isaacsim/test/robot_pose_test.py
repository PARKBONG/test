import os
import cv2
import math
import random
import numpy as np
from isaacgym import gymapi, gymutil
import torch as T

device = "cuda:0" if T.cuda.is_available() else "cpu"

# Color
env_color = gymapi.Vec3(247.0/255.0, 248.0/255.0, 249.0/255.0)      # white
robot_color = gymapi.Vec3(222.0/255.0, 122.0/255.0, 117.0/255.0)    # Red
obstacle_color = gymapi.Vec3(44.0/255.0, 74.0/255.0, 96.0/255.0)    # dark blue

# Initialize Gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()

# create a simulator
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0

# physx variables
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    # sim_params.physx.rest_offset = 0.001
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# GPU Tensor를 사용하려면 use_gpu_pipeline를 True로 설정해야함
sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = True
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# Create Simulate
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")
    quit()

# 모든 환경이 완전히 설정 prepare_sim되면 Tensor API에서 사용하는 내부 데이터 구조를 초기화하기 위해 호출해야 합니다.
gym.prepare_sim(sim)

# Create Viewer
camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0
camera_props.height = 640
camera_props.width = 840
camera_handle = gym.create_viewer(sim, camera_props)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)    # set the normal force to be z dimension
plane_params.static_friction = 0.7
plane_params.dynamic_friction = 0.7
plane_params.restitution = 0.0
# plane_params.ground_texture = "../textures/texture_background_wall_paint_3.jpg"
gym.add_ground(sim, plane_params)

# Asset Description
asset_root = "./urdf"
robot_asset_file = "/ugv_gazebo_sim/scout/scout_description/urdf/scout_mini.urdf"
env_asset_file = "base_link_description/urdf/base_link.urdf"
obstacle_asset_file = "/box_obstacle_description/urdf/box_obstacle.urdf"
# Load texture from file
texture_file = "./textures/texture_background_wall_paint_3.jpg"
texture_handle = gym.create_texture_from_file(sim, os.path.join("", texture_file))

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
asset_options.replace_cylinder_with_capsule = True

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

# Robot Dof
robot_num_dof = gym.get_asset_dof_count(robot_asset)
robot_dof_names = gym.get_asset_dof_names(robot_asset)
robot_dof_name_to_id = {k: v for k, v in zip(robot_dof_names, np.arange(robot_num_dof))}
print("number of asset: {}, dof names: {}".format(robot_num_dof, robot_dof_names))
print("robot_dof_name_to_id: {}".format(robot_dof_name_to_id))

# Create Environments
print("Creating {} environment".format(num_envs))
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)

    # Environment Generate
    env_pose.p = gymapi.Vec3(0.69, 0.02, 0)
    env_handle = gym.create_actor(env, env_asset, env_pose, "bin", i, 0)
    gym.set_rigid_body_color(env, env_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, env_color)

    # Obstacle Generate
    
    # obstacle_pose.p = gymapi.Vec3(1.5, 0.4, 0.0)
    # obstacle_handle = gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i ,0) # This create obstacle

    # Actor Generate
    robot_init = [0,0,0]
    robot_pose.p = gymapi.Vec3(robot_init[0], robot_init[1], robot_init[2])
    robot_handle = gym.create_actor(env, robot_asset, robot_pose, "Scout_Mini", i, 0)
    gym.set_rigid_body_color(env, robot_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, robot_color)
    # gym.set_rigid_body_texture(env, robot_handle, 0, gymapi.MESH_VISUAL, env_floor) # NO ENV_floor -> commented out

    # Set controller modes
    props = gym.get_actor_dof_properties(env, robot_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    props["stiffness"] = (0.0, 0.0, 0.0, 0.0)   # DOF_MODE_VEL 속도제어에 사용되며 stiffness variable은 0으로 설정해야 함 (PD controller에 의해 적용되는 토크는 damping parameter에 비례한다.)
    props["damping"] = (6000.0, 6000.0, 6000.0, 6000.0)
    gym.set_actor_dof_properties(env, robot_handle, props)
    vel_targets = np.random.uniform(-2*math.pi, 2*math.pi, robot_num_dof).astype('f')
    # set_actor_dof_velocity_target를 사용해서 속도 목표를 설정할 수 있음.
    # DOF가 linear인 경우 목표는 초당 미터기이다.
    # DOF가 angular인 경우 목표는 초당 라디안이다.
    # Unlike efforts, position and velocity targets don’t need to be set every frame, only when changing targets.
    gym.set_actor_dof_velocity_targets(env, robot_handle, (-2*math.pi, 2*math.pi, -2*math.pi, 2*math.pi))
    # gym.set_actor_dof_velocity_targets(env, robot_handle, vel_targets)

    # Option 1 =======================================================================================
    # robot_size = 0.2
    # def gen_obstacle_handle(randomness=True):
            
    #     def find_value(box_size):

    #         x_random = np.random.uniform(box_size[0], box_size[1])
    #         z_random = np.random.uniform(box_size[2], box_size[3])
    #         if ((robot_init[0] - robot_size) < x_random) and (x_random < (robot_init[0] + robot_size)):
    #             if ((robot_init[2] - robot_size) < z_random) and (z_random < (robot_init[2] + robot_size)):
    #                     x_random, z_random = find_value(box_size)
    #         return x_random, z_random

    #     if randomness:
    #         box_size = [1, 2, -1, 1] #[x_min, x_max, z_min, z_max] # Find your best value white big box
    #         x_random, z_random = find_value(box_size)
    #         obstacle_pose.p = gymapi.Vec3(x_random, 0.4, z_random)
    #     else : 
    #         obstacle_pose.p = gymapi.Vec3(.5, 0.4, 0.0)

    #     obstacle_handle = gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i ,0) # This create obstacle
    #     return obstacle_handle

    # n_obstacles = 2
    # obstacle_handle_list = [gen_obstacle_handle() for i in range(n_obstacles)]

    # Option 2 =====================================================================================
    # White Box Size # Change properly!
    x_min = -5
    x_max = 5
    z_min = -5
    z_max = 5

    # Robot init position # DO not change
    x_list = [0]
    z_list = [0]

    norm = 1 # Car or Obstacle size # Change properly!

    def get_norm1(x1,z1,x2,z2):
        norm = np.sqrt((x1 - x2)**2 + (z1 - z2)**2)
        return norm

    def get_random(x_list,z_list):
        x_random = np.random.uniform(x_min, x_max)
        z_random = np.random.uniform(z_min, z_max)

        for x,z in zip(x_list,z_list):
            if get_norm1(x,z,x_random,z_random) < norm: 
                x_random, z_random = get_random(x_list,z_list)
        return x_random, z_random

    n = 12
    for i in range(n):
        x_random, z_random = get_random(x_list,z_list)
        x_list.append(x_random)
        z_list.append(z_random)

    obstacle_handle_list = []
    x_list.pop(0) # Car position
    z_list.pop(0)
    for x,z in zip(x_list,z_list):
        obstacle_pose.p = gymapi.Vec3(x, z, 0.4) # height = 0.4
        obstacle_handle = gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i ,0) # This create obstacle
        obstacle_handle_list.append(obstacle_handle)

    # Set Color ========================================================================================
    for obstacle_handle in obstacle_handle_list:
        gym.set_rigid_body_color(env, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color)

# Camera Setting
cam_pos = gymapi.Vec3(1.0, 0.0, 1.0)      # 멀리 보려면 z축 변경 가까이 보려면 x축 변경
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(camera_handle, None, cam_pos, cam_target)

# Set up text objects for OpenCV
font = cv2.FONT_HERSHEY_SIMPLEX
font_thickness = 2
font_scale = 0.8
font_pos = (10, 30)
font_color = (0, 255, 0)

# video_writer = cv2.VideoWriter("../video/output.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, (WIDTH, HEIGHT))

# Simulate
while True:
    # check if we should update
    t = gym.get_sim_time(sim)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    vel_targets = np.random.uniform(-2*math.pi, 2*math.pi, robot_num_dof).astype('f')
    gym.set_actor_dof_velocity_targets(env, robot_handle, (-2*math.pi, 2*math.pi, -2*math.pi, 2*math.pi))

    # update graphics transforms
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # render the viewer
    gym.draw_viewer(camera_handle, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    # Check for exit condition - user closed the viewer window
    if gym.query_viewer_has_closed(camera_handle):
        break

print('Done')

gym.destroy_viewer(viewer)

gym.destroy_sim(sim)