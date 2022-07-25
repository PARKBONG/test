from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Convex decomposition example")

# create a simulator
sim_params = gymapi.SimParams()

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

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

# set up the env grid
num_envs = 4
envs_per_row = 2
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# initial root pose for actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 0.2)

asset_root = "../urdf"
asset_options = gymapi.AssetOptions()

# Load materials from meshes
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

# Override the bogus inertia tensors and center-of-mass properties in the YCB assets.
# These flags will force the inertial properties to be recomputed from geometry.
asset_options.override_inertia = True
asset_options.override_com = True

# use default convex decomposition params
asset_options.vhacd_enabled = True
asset0 = gym.load_asset(sim, asset_root, "/box_obstacle_description/urdf/box_obstacle.urdf", asset_options)

# create envs
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
actor = gym.create_actor(env, asset0, initial_pose, 'actor', 0, 1)

cam_pos = gymapi.Vec3(-5, 0, 4)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
while not gym.query_viewer_has_closed(viewer):

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