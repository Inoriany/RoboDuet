"""
gen_grab_object.py  v2
=======================
Fixed-base arm reaching toward a physical box in the scene.

The key challenge: IsaacGym requires all actors to be added BEFORE
prepare_sim() is called.  We subclass VelocityTrackingEasyEnv and
override _create_envs() to insert a coloured static box alongside
the robot inside the env-creation loop.

Usage (on server):
    python /root/RoboDuet/gen_grab_object.py
"""

# CRITICAL: isaacgym BEFORE torch
import isaacgym
assert isaacgym
import torch
import math
import os
import sys
import time

sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi
import imageio

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch import global_switch

# ─── Paths ────────────────────────────────────────────────────────────────────
ARM_CKPT = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_arm/ac_weights_last_arm.pt"
)
VIDEO_OUT = "/root/RoboDuet/b2z1_grab_object.mp4"

# ─── Arm target (spherical coords, body frame) ────────────────────────────────
# l in [0.3, 0.77] m  |  p (pitch up) in [-0.45pi, 0.45pi]  |  y (yaw) in [-pi/2, pi/2]
CMD_L = 0.55          # 0.55 m reach
CMD_P = 0.25          # ~14 deg upward
CMD_Y = 0.0           # straight ahead

# Target world position (robot starts at origin, base_z ~0.44m, no yaw)
# _lpy_to_world_xyz: z_ = l*sin(p) + measured_heights + 0.38  (flat plane -> mh~0)
BOX_X = CMD_L * math.cos(CMD_P) * math.cos(CMD_Y)   # ~0.533
BOX_Y = CMD_L * math.cos(CMD_P) * math.sin(CMD_Y)   # ~0.000
BOX_Z = CMD_L * math.sin(CMD_P) + 0.38              # ~0.516
BOX_HALF = 0.05   # 10 cm cube (visible but not blocking arm)

NUM_STEPS    = 750   # 25 s at 30 fps
WARMUP_STEPS = 80
FPS          = 30


# ─── Config ───────────────────────────────────────────────────────────────────
def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    KP, KD = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP
    Cfg.dog.control.damping_leg["joint"]   = KD
    Cfg.control.stiffness["joint"]         = KP
    Cfg.control.damping["joint"]           = KD

    Cfg.env.keep_arm_fixed = False
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0

    Cfg.commands.lin_vel_x    = [0.0, 0.0]
    Cfg.commands.lin_vel_y    = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw  = [0.0, 0.0]
    Cfg.commands.limit_vel_x  = [0.0, 0.0]
    Cfg.commands.limit_vel_y  = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]

    Cfg.commands.distributional_commands     = False
    Cfg.domain_rand.lag_timesteps            = 6
    Cfg.domain_rand.randomize_lag_timesteps  = False
    Cfg.control.control_type                 = "M"
    Cfg.domain_rand.added_mass_range         = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions         = False
    Cfg.commands.body_roll_range             = [-0.4, 0.4]
    Cfg.commands.limit_body_roll             = [-0.4, 0.4]
    Cfg.commands.body_pitch_range            = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch            = [-0.4, 0.4]
    Cfg.env.num_envs                         = 1
    Cfg.terrain.mesh_type                    = "plane"
    Cfg.terrain.teleport_robots              = False
    Cfg.control.update_obs_freq              = 20
    Cfg.env.num_actions                      = 18
    Cfg.env.num_observations                 = 63

    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_energy       = -0.00004
    Cfg.reward_scales.loco_energy             = -0.00004
    Cfg.reward_scales.jump                    = 0.0
    Cfg.rewards.terminal_body_height          = 0.05
    Cfg.rewards.use_terminal_body_height      = False
    Cfg.env.max_episode_length                = 99999

    Cfg.domain_rand.randomize_friction         = False
    Cfg.domain_rand.randomize_base_mass        = False
    Cfg.domain_rand.randomize_restitution      = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength   = False
    Cfg.domain_rand.randomize_motor_offset     = False
    Cfg.domain_rand.randomize_gravity          = False
    Cfg.domain_rand.push_robots                = False

    Cfg.commands.T_force_range               = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres             = 0.3
    Cfg.domain_rand.max_force                = 15
    Cfg.domain_rand.max_force_offset         = 0.01
    Cfg.env.priv_observe_vel                 = False
    Cfg.commands.global_reference            = False
    Cfg.env.priv_observe_high_freq_goal      = False
    Cfg.dog.dog_num_privileged_obs           = 2
    Cfg.arm.arm_num_privileged_obs           = 9
    Cfg.env.num_privileged_obs               = 9
    Cfg.asset.render_sphere                  = True
    Cfg.hybrid.use_vision                    = False
    Cfg.rewards.manip_weight_lpy             = 3
    Cfg.rewards.manip_weight_rpy             = 1
    Cfg.hybrid.reward_scales.arm_dof_vel     = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc     = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1)
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2)
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    global_switch.init_sigmoid_lr()


# ─── Subclass: add box during _create_envs ────────────────────────────────────
class EnvWithBox(VelocityTrackingEasyEnv):
    """Subclass that adds a static coloured box alongside the robot during env creation."""

    box_pos  = (BOX_X, BOX_Y, BOX_Z)
    box_half = BOX_HALF

    def _create_envs(self):
        # Run normal env creation first (adds robot)
        super()._create_envs()

        # Now add a static box to each env
        bx, by, bz = self.box_pos
        hs = self.box_half
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True    # static
        asset_opts.density       = 1000.0
        asset_opts.disable_gravity = True

        box_asset = self.gym.create_box(self.sim, hs * 2, hs * 2, hs * 2, asset_opts)

        for i, env_handle in enumerate(self.envs):
            # Place box at world position (env_origin + offset)
            origin = self.env_origins[i]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                origin[0].item() + bx,
                origin[1].item() + by,
                bz)
            pose.r = gymapi.Quat(0, 0, 0, 1)

            # collision group 2 -> no collision with robot (group 0)
            box_handle = self.gym.create_actor(
                env_handle, box_asset, pose, "target_box", i, 2)

            # Bright red-orange
            color = gymapi.Vec3(1.0, 0.25, 0.0)
            self.gym.set_rigid_body_color(
                env_handle, box_handle, 0, gymapi.MESH_VISUAL, color)

        print(f"[EnvWithBox] Added box at ({bx:.3f}, {by:.3f}, {bz:.3f}) "
              f"to {len(self.envs)} env(s)")


def fix_arm_commands(env):
    env.commands_arm[:, 0] = CMD_L
    env.commands_arm[:, 1] = CMD_P
    env.commands_arm[:, 2] = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = CMD_P
    env.commands_arm_obs[:, 2] = CMD_Y


def main():
    print("=" * 60)
    print("  B2Z1 Fixed-Base Arm -> Physical Box")
    print(f"  CMD: l={CMD_L}, p={CMD_P:.3f}rad ({math.degrees(CMD_P):.1f}deg)")
    print(f"  Box world pos: ({BOX_X:.3f}, {BOX_Y:.3f}, {BOX_Z:.3f})")
    print("=" * 60)

    setup_config()
    global_switch.open_switch()
    print(f"[Config] switch_open={global_switch.switch_open}, "
          f"keep_arm_fixed={Cfg.env.keep_arm_fixed}")

    # ── Environment with embedded box ─────────────────────────────────────────
    print("\n[1/4] Creating environment (with embedded target box) ...")
    env = EnvWithBox(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)
    print("[OK]  Environment ready")

    # ── Camera ────────────────────────────────────────────────────────────────
    cp = gymapi.CameraProperties()
    cp.width = 1280; cp.height = 720; cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)
    env.gym.set_camera_location(
        cam, env.envs[0],
        gymapi.Vec3(1.5, 0.9, 0.9),
        gymapi.Vec3(0.3, 0.0, 0.4))

    # ── Arm policy ────────────────────────────────────────────────────────────
    print("\n[2/4] Loading arm policy ...")
    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)  # 8-6=2
    print(f"[OK]  Arm policy (obs={Cfg.arm.arm_num_observations}, "
          f"hist={Cfg.arm.arm_num_obs_history}, "
          f"act={Cfg.arm.num_actions_arm_cd}, plan_dims={num_plan})")

    # ── Reset ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Reset + set fixed arm commands ...")
    env.reset()
    env.commands_dog[:, :] = 0.0
    fix_arm_commands(env)
    print("[OK]  Reset done")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"[3.5/4] Warmup ({WARMUP_STEPS} steps) ...")
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            fix_arm_commands(env)
            arm_obs = env.get_arm_observations()
            arm_policy.update_distribution(arm_obs['obs_history'])
            arm_full = arm_policy.action_mean
            if num_plan > 0:
                env.plan(arm_full[..., -num_plan:])
            action_arm = arm_full[..., :-num_plan] if num_plan > 0 else arm_full
            action_dog = torch.zeros((1, 12), device='cuda:0')
            env.step(action_dog, action_arm)
            env.commands_dog[:, :] = 0.0
    print("[OK]  Warmup done")

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"\n[4/4] Running {NUM_STEPS} steps ...")
    frames = []
    t0 = time.time()

    with torch.no_grad():
        for step in range(NUM_STEPS):
            fix_arm_commands(env)

            arm_obs = env.get_arm_observations()
            arm_policy.update_distribution(arm_obs['obs_history'])
            arm_full = arm_policy.action_mean
            if num_plan > 0:
                env.plan(arm_full[..., -num_plan:])
            action_arm = arm_full[..., :-num_plan] if num_plan > 0 else arm_full
            action_dog = torch.zeros((1, 12), device='cuda:0')
            env.step(action_dog, action_arm)
            env.commands_dog[:, :] = 0.0

            # Camera follows robot but keeps box visible
            rx = env.root_states[0, 0].item()
            ry = env.root_states[0, 1].item()
            rz = env.root_states[0, 2].item()
            env.gym.set_camera_location(
                cam, env.envs[0],
                gymapi.Vec3(rx + 1.5, ry + 0.9, rz + 0.6),
                gymapi.Vec3(rx + 0.25, ry, rz + 0.15))
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(
                env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
            img = img.reshape(cp.height, cp.width, 4)[:, :, :3]
            frames.append(img)

            if step == 0 or (step + 1) % 150 == 0:
                fps_now = (step + 1) / max(time.time() - t0, 1e-6)
                print(f"  Step {step+1:>4d}/{NUM_STEPS}  "
                      f"h={rz:.3f}  {fps_now:.1f}fps")

    elapsed = time.time() - t0
    print(f"\nDone: {NUM_STEPS} steps in {elapsed:.1f}s "
          f"({NUM_STEPS/elapsed:.1f} fps)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving {len(frames)} frames -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
