"""
gen_grasp_lift_v8.py
====================
Stable presentation demo:
- zero dog actions
- scripted 6-joint arm motion
- gripper closed by direct DOF state write
- box starts at a fixed grasp point
- short smooth attach phase, then lifted with the gripper
"""

import isaacgym
assert isaacgym
import torch
import sys
import time

sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi, gymtorch
import imageio

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym.utils.global_switch import global_switch

VIDEO_OUT = "/root/RoboDuet/b2z1_grasp_lift_v8.mp4"
FPS = 30
BOX_HALF = 0.025

WARMUP_STEPS = 60
HOLD0_STEPS = 40
REACH_STEPS = 120
CLOSE_STEPS = 45
ATTACH_STEPS = 20
LIFT_STEPS = 120
HOLD2_STEPS = 60
NUM_STEPS = HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + ATTACH_STEPS + LIFT_STEPS + HOLD2_STEPS

ARM_START = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device='cuda:0')
ARM_REACH = torch.tensor([0.0, -4.0, 4.0, 0.0, 0.0, 0.0], device='cuda:0')
ARM_LIFT = torch.tensor([0.0, -3.9, 3.9, 0.0, 0.0, 0.0], device='cuda:0')

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -0.80

# Fixed box position from the stable no-box probe at the reach pose.
BOX_START = torch.tensor([-0.088, -0.162, 0.859], device='cuda:0')
ATTACH_OFFSET = torch.tensor([0.0, 0.0, -BOX_HALF * 0.35], device='cuda:0')


def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    kp, kd = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = kp
    Cfg.dog.control.damping_leg["joint"] = kd
    Cfg.control.stiffness["joint"] = kp
    Cfg.control.damping["joint"] = kd

    Cfg.env.keep_arm_fixed = False
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end = 0

    Cfg.commands.lin_vel_x = [0.0, 0.0]
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    Cfg.commands.limit_vel_x = [0.0, 0.0]
    Cfg.commands.limit_vel_y = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]
    Cfg.commands.distributional_commands = False

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "M"
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    Cfg.env.num_envs = 1
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63
    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy = -0.00004
    Cfg.reward_scales.jump = 0.0
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_body_height = False
    Cfg.env.max_episode_length = 99999
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.push_robots = False
    Cfg.commands.T_force_range = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres = 0.3
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01
    Cfg.env.priv_observe_vel = False
    Cfg.commands.global_reference = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9
    Cfg.asset.render_sphere = True
    Cfg.hybrid.use_vision = False
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1
    Cfg.hybrid.reward_scales.arm_dof_vel = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * Cfg.reward_scales.action_smoothness_1
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * Cfg.reward_scales.action_smoothness_2
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    global_switch.init_sigmoid_lr()


class EnvWithBox(VelocityTrackingEasyEnv):
    def _create_envs(self):
        super()._create_envs()
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.density = 500.0
        opts.disable_gravity = True
        box_asset = self.gym.create_box(self.sim, BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2, opts)
        for i, env_handle in enumerate(self.envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(BOX_START[0].item(), BOX_START[1].item(), BOX_START[2].item())
            pose.r = gymapi.Quat(0, 0, 0, 1)
            bh = self.gym.create_actor(env_handle, box_asset, pose, "target_box", i, 2)
            self.gym.set_rigid_body_color(env_handle, bh, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.25, 0.0))


def lerp(a, b, t):
    return a + (b - a) * t


def set_gripper(env, value):
    env.dof_pos[:, 18] = value
    env.dof_vel[:, 18] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def step_env(env, arm_action, gripper_value):
    dog_action = torch.zeros((1, 12), device='cuda:0')
    env.step(dog_action, arm_action.unsqueeze(0))
    env.commands_dog[:, :] = 0.0
    set_gripper(env, gripper_value)


def move_box(env, pos):
    env.root_states[1, 0:3] = pos
    env.root_states[1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda:0')
    env.root_states[1, 7:13] = 0.0
    idx = torch.tensor([1], dtype=torch.int32, device='cuda:0')
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(idx),
        1,
    )


def get_gripper_mover_pos(env):
    bodies = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
    return bodies[0, 25, :3].clone()


def capture(env, cam, cp):
    rx = env.root_states[0, 0].item()
    ry = env.root_states[0, 1].item()
    rz = env.root_states[0, 2].item()
    env.gym.set_camera_location(
        cam,
        env.envs[0],
        gymapi.Vec3(rx + 1.5, ry + 0.9, rz + 0.6),
        gymapi.Vec3(rx - 0.05, ry - 0.10, rz + 0.45),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def main():
    print("=" * 60)
    print("  B2Z1 Grasp Lift v8  (stable scripted grasp demo)")
    print(f"  box start = ({BOX_START[0]:.3f}, {BOX_START[1]:.3f}, {BOX_START[2]:.3f})")
    print("=" * 60)

    setup_config()
    global_switch.open_switch()

    print("[1/3] Creating env ...")
    env = EnvWithBox(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)

    cp = gymapi.CameraProperties()
    cp.width = 1280
    cp.height = 720
    cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    print("[2/3] Warmup ...")
    env.reset()
    env.commands_dog[:, :] = 0.0
    move_box(env, BOX_START)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            step_env(env, ARM_START, GRIPPER_OPEN)
        move_box(env, BOX_START)

    print("[3/3] Recording ...")
    frames = []
    t0 = time.time()
    attached = False
    box_pos = BOX_START.clone()

    with torch.no_grad():
        for step in range(NUM_STEPS):
            if step < HOLD0_STEPS:
                arm = ARM_START
                grip = GRIPPER_OPEN
                phase = "HOLD0"
            elif step < HOLD0_STEPS + REACH_STEPS:
                frac = (step - HOLD0_STEPS + 1) / REACH_STEPS
                arm = lerp(ARM_START, ARM_REACH, frac)
                grip = GRIPPER_OPEN
                phase = "REACH"
            elif step < HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS:
                frac = (step - HOLD0_STEPS - REACH_STEPS + 1) / CLOSE_STEPS
                arm = ARM_REACH
                grip = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, frac)
                phase = "CLOSE"
            elif step < HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + ATTACH_STEPS:
                arm = ARM_REACH
                grip = GRIPPER_CLOSED
                phase = "ATTACH"
            elif step < HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + ATTACH_STEPS + LIFT_STEPS:
                frac = (step - HOLD0_STEPS - REACH_STEPS - CLOSE_STEPS - ATTACH_STEPS + 1) / LIFT_STEPS
                arm = lerp(ARM_REACH, ARM_LIFT, frac)
                grip = GRIPPER_CLOSED
                phase = "LIFT"
            else:
                arm = ARM_LIFT
                grip = GRIPPER_CLOSED
                phase = "HOLD2"

            step_env(env, arm, grip)

            if phase == "ATTACH":
                gm = get_gripper_mover_pos(env) + ATTACH_OFFSET
                frac = (step - HOLD0_STEPS - REACH_STEPS - CLOSE_STEPS + 1) / ATTACH_STEPS
                box_pos = lerp(BOX_START, gm, frac)
                move_box(env, box_pos)
                attached = frac >= 1.0
            elif phase in ("LIFT", "HOLD2"):
                gm = get_gripper_mover_pos(env) + ATTACH_OFFSET
                box_pos = gm
                move_box(env, box_pos)
                attached = True

            frames.append(capture(env, cam, cp))

            if step == 0 or (step + 1) % 60 == 0:
                h = env.root_states[0, 2].item()
                fps_now = (step + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  Step {step+1:>4d}/{NUM_STEPS}  h={h:.3f}  "
                    f"box=({box_pos[0]:.3f},{box_pos[1]:.3f},{box_pos[2]:.3f})  "
                    f"attach={attached}  [{phase}]  {fps_now:.1f}fps"
                )

    elapsed = time.time() - t0
    print(f"Done: {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS/elapsed:.1f}fps)")
    print(f"Saving -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
