"""
test_grasp.py -- Video generation for grasp_stand checkpoint
=============================================================
Loads both dog + arm checkpoints from b2z1_grasp_stand,
runs stand-still + arm-manipulation inference, saves .mp4.

Usage (on server, inside conda roboduet):
    python /root/RoboDuet/test_grasp.py
    python /root/RoboDuet/test_grasp.py --num_steps 600
    python /root/RoboDuet/test_grasp.py \\
        --dog_ckpt /path/to/dog.pt --arm_ckpt /path/to/arm.pt
"""

# ============================================================
# CRITICAL: isaacgym MUST come before torch
# ============================================================
import isaacgym
assert isaacgym
import torch
import numpy as np
import os
import sys
import glob
import argparse
import time

sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch import global_switch

RUNS_ROOT = "/root/RoboDuet/runs"


# ============================================================
# Helpers: auto-detect latest checkpoints
# ============================================================
def find_latest(run_name, subdir, filename="ac_weights_last*.pt"):
    pattern = os.path.join(RUNS_ROOT, run_name, "**", subdir, filename)
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        # fall back to any numbered checkpoint
        pattern2 = os.path.join(RUNS_ROOT, run_name, "**", subdir, "ac_weights_*.pt")
        candidates = glob.glob(pattern2, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


# ============================================================
# Config -- mirrors auto_train_grasp.py EXACTLY
# ============================================================
def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    # Correct PD gains
    KP_LEG, KD_LEG = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP_LEG
    Cfg.dog.control.damping_leg["joint"]   = KD_LEG
    Cfg.control.stiffness["joint"]         = KP_LEG
    Cfg.control.damping["joint"]           = KD_LEG

    # ARM ENABLED from step 0
    Cfg.env.keep_arm_fixed = False
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0

    # Zero locomotion commands (dog stands still)
    Cfg.commands.lin_vel_x    = [0.0, 0.0]
    Cfg.commands.lin_vel_y    = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw  = [0.0, 0.0]
    Cfg.commands.limit_vel_x  = [0.0, 0.0]
    Cfg.commands.limit_vel_y  = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]

    Cfg.commands.distributional_commands = False
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "M"
    Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range  = [-0.4, 0.4]
    Cfg.commands.limit_body_roll  = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]

    Cfg.env.num_envs = 1
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions      = 18
    Cfg.env.num_observations = 63

    # Reward scales (read by env init even during inference)
    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_energy   = -0.00004
    Cfg.reward_scales.loco_energy         = -0.00004
    Cfg.reward_scales.jump                = -0.00
    # Disable fall-reset for cleaner video
    Cfg.rewards.terminal_body_height      = 0.05
    Cfg.rewards.use_terminal_body_height  = False
    Cfg.env.max_episode_length            = 99999

    # No domain randomisation during eval
    Cfg.domain_rand.randomize_friction         = False
    Cfg.domain_rand.randomize_base_mass        = False
    Cfg.domain_rand.randomize_restitution      = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength   = False
    Cfg.domain_rand.randomize_motor_offset     = False
    Cfg.domain_rand.randomize_gravity          = False
    Cfg.domain_rand.push_robots                = False

    Cfg.commands.T_force_range             = [2, 4.]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres           = 0.3
    Cfg.domain_rand.max_force              = 15
    Cfg.domain_rand.max_force_offset       = 0.01
    Cfg.env.priv_observe_vel               = False
    Cfg.commands.global_reference          = False
    Cfg.env.priv_observe_high_freq_goal    = False
    Cfg.dog.dog_num_privileged_obs         = 2
    Cfg.arm.arm_num_privileged_obs         = 9
    Cfg.env.num_privileged_obs             = 9
    Cfg.asset.render_sphere                = True
    Cfg.hybrid.use_vision                  = False
    Cfg.rewards.manip_weight_lpy           = 3
    Cfg.rewards.manip_weight_rpy           = 1
    Cfg.hybrid.reward_scales.arm_dof_vel   = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc   = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = (
        10 * Cfg.reward_scales.action_rate)
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1)
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2)
    Cfg.use_rot6d = False
    Cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    )
    global_switch.init_sigmoid_lr()


# ============================================================
# Camera helpers
# ============================================================
def create_recording_camera(gym, sim, envs, width=1280, height=720):
    camera_props = gymapi.CameraProperties()
    camera_props.width  = width
    camera_props.height = height
    camera_props.enable_tensors = False
    camera_handle = gym.create_camera_sensor(envs[0], camera_props)
    # Slightly wider view to capture arm motion
    cam_pos    = gymapi.Vec3(1.2, 0.8, 0.6)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.3)
    gym.set_camera_location(camera_handle, envs[0], cam_pos, cam_target)
    return camera_handle, camera_props


def update_camera(gym, envs, camera_handle, root_states):
    rx = root_states[0, 0].item()
    ry = root_states[0, 1].item()
    rz = root_states[0, 2].item()
    cam_pos    = gymapi.Vec3(rx + 1.2, ry + 0.8, rz + 0.4)
    cam_target = gymapi.Vec3(rx, ry, rz + 0.1)
    gym.set_camera_location(camera_handle, envs[0], cam_pos, cam_target)


def capture_frame(gym, sim, envs, camera_handle, camera_props, root_states=None):
    if root_states is not None:
        update_camera(gym, envs, camera_handle, root_states)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = gym.get_camera_image(sim, envs[0], camera_handle, gymapi.IMAGE_COLOR)
    img = img.reshape(camera_props.height, camera_props.width, 4)
    return img[:, :, :3]


def save_video(frames, path, fps=30):
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[OK] Video saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return
    except ImportError:
        pass
    try:
        import cv2
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[OK] Video saved (cv2): {path}")
        return
    except ImportError:
        pass
    frames_dir = path.replace('.mp4', '_frames')
    os.makedirs(frames_dir, exist_ok=True)
    for i, f in enumerate(frames):
        ppm = os.path.join(frames_dir, f"frame_{i:04d}.ppm")
        h, w = f.shape[:2]
        with open(ppm, 'wb') as fp:
            fp.write(f"P6\n{w} {h}\n255\n".encode())
            fp.write(f.tobytes())
    print(f"[OK] Saved {len(frames)} frames to {frames_dir}/")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test grasp_stand checkpoint - stand + arm manipulation video")
    parser.add_argument("--dog_ckpt", type=str, default=None,
                        help="Dog checkpoint path (auto-detected if omitted)")
    parser.add_argument("--arm_ckpt", type=str, default=None,
                        help="Arm checkpoint path (auto-detected if omitted)")
    parser.add_argument("--run_name", type=str, default="b2z1_grasp_stand")
    parser.add_argument("--num_steps", type=int, default=600,
                        help="Number of simulation steps (default 600 = 20s @ 30fps)")
    parser.add_argument("--video_path", type=str,
                        default="/root/RoboDuet/b2z1_grasp_video.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_steps", type=int, default=50)
    args = parser.parse_args()

    # ---- Find checkpoints -----------------------------------------------
    dog_ckpt = args.dog_ckpt or find_latest(args.run_name, "checkpoints_dog")
    arm_ckpt = args.arm_ckpt or find_latest(args.run_name, "checkpoints_arm")
    print(f"[DOG CKPT] {dog_ckpt}")
    print(f"[ARM CKPT] {arm_ckpt}")

    # ---- Config --------------------------------------------------------
    setup_config()
    # Open the hybrid switch so arm actions are used
    global_switch.open_switch()
    print(f"[CONFIG] keep_arm_fixed={Cfg.env.keep_arm_fixed}  "
          f"switch_open={global_switch.switch_open}")

    # ---- Environment ---------------------------------------------------
    print("\n[1/4] Creating environment ...")
    env = VelocityTrackingEasyEnv(
        sim_device="cuda:0", headless=True, num_envs=1, cfg=Cfg
    )
    env = HistoryWrapper(env)
    print("[OK]  Environment ready")

    # ---- Camera --------------------------------------------------------
    print("[+] Setting up recording camera ...")
    camera_handle, camera_props = create_recording_camera(
        env.gym, env.sim, env.envs
    )

    # ---- Dog Policy ----------------------------------------------------
    print("\n[2/4] Loading dog policy ...")
    dog_obs_hist_dim = Cfg.dog.dog_num_obs_history
    dog_priv_obs_dim = Cfg.dog.dog_num_privileged_obs
    dog_action_dim   = 12
    num_arm_step     = Cfg.env.num_actions - dog_action_dim    # 6
    num_plan_actions = Cfg.arm.num_actions_arm_cd - num_arm_step  # 8 - 6 = 2

    dog_policy = DogActorCritic(
        num_obs=1,
        num_privileged_obs=dog_priv_obs_dim,
        num_obs_history=dog_obs_hist_dim,
        num_actions=dog_action_dim,
    ).to("cuda:0")
    dog_policy.load_state_dict(torch.load(dog_ckpt, map_location="cuda:0"))
    dog_policy.eval()
    print(f"[OK]  Dog policy loaded")

    # ---- Arm Policy ----------------------------------------------------
    print("[3a/4] Loading arm policy ...")
    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to("cuda:0")
    arm_policy.load_state_dict(torch.load(arm_ckpt, map_location="cuda:0"))
    arm_policy.eval()
    print(f"[OK]  Arm policy loaded  "
          f"(obs={Cfg.arm.arm_num_observations}, "
          f"obs_hist={Cfg.arm.arm_num_obs_history}, "
          f"actions={Cfg.arm.num_actions_arm_cd})")

    # ---- Reset + zero commands ----------------------------------------
    print("\n[3/4] Resetting environment ...")
    env.reset()
    env.commands_dog[:, 0] = 0.0    # vx = 0 (stand still)
    env.commands_dog[:, 1:5] = 0.0

    # Pre-fill dog obs history
    dog_num_obs = Cfg.dog.dog_num_observations
    dog_slots   = dog_obs_hist_dim // dog_num_obs
    first_dog   = env.get_dog_observations()
    env.dog_obs_history[:] = first_dog["obs"].repeat(1, dog_slots)

    # ---- Warmup --------------------------------------------------------
    print(f"[3.5/4] Warmup ({args.warmup_steps} steps) ...")
    with torch.no_grad():
        for ws in range(args.warmup_steps):
            arm_obs  = env.get_arm_observations()
            arm_policy.update_distribution(arm_obs["obs_history"])
            arm_actions_full = arm_policy.action_mean        # [1, 8]
            if num_plan_actions > 0:
                env.plan(arm_actions_full[..., -num_plan_actions:])
            action_arm = arm_actions_full[..., :-num_plan_actions] if num_plan_actions > 0 else arm_actions_full

            dog_obs  = env.get_dog_observations()
            action_d = dog_policy.act_teacher(
                dog_obs["obs_history"], dog_obs["privileged_obs"])

            _, _, done, _ = env.step(action_d, action_arm)
            env.commands_dog[:, 0] = 0.0
            env.commands_dog[:, 1:5] = 0.0
            if done.any():
                env.dog_obs_history[:] = (
                    env.get_dog_observations()["obs"].repeat(1, dog_slots))
    print("[OK]  Warmup done")

    # ---- Inference loop ------------------------------------------------
    print(f"\n[4/4] Running {args.num_steps} steps (stand + arm manip) ...")
    frames = []
    total_rew_dog = 0.0
    total_rew_arm = 0.0
    resets = 0
    t0 = time.time()

    with torch.no_grad():
        for step in range(args.num_steps):
            # Arm policy inference
            arm_obs  = env.get_arm_observations()
            arm_policy.update_distribution(arm_obs["obs_history"])
            arm_actions_full = arm_policy.action_mean        # [1, 8]
            if num_plan_actions > 0:
                env.plan(arm_actions_full[..., -num_plan_actions:])
            action_arm = arm_actions_full[..., :-num_plan_actions] if num_plan_actions > 0 else arm_actions_full

            # Dog policy inference (teacher mode)
            dog_obs  = env.get_dog_observations()
            action_d = dog_policy.act_teacher(
                dog_obs["obs_history"], dog_obs["privileged_obs"])

            rew_d, rew_a, done, _ = env.step(action_d, action_arm)
            total_rew_dog += rew_d.item()
            total_rew_arm += rew_a.item()

            env.commands_dog[:, 0] = 0.0
            env.commands_dog[:, 1:5] = 0.0

            # Capture frame
            frame = capture_frame(env.gym, env.sim, env.envs,
                                  camera_handle, camera_props,
                                  root_states=env.root_states[:1])
            frames.append(frame)

            if (step + 1) % 100 == 0 or step == 0:
                elapsed = time.time() - t0
                fps_cur = (step + 1) / max(elapsed, 1e-6)
                body_h  = env.root_states[0, 2].item()
                print(f"  Step {step+1:>4d}/{args.num_steps}  "
                      f"dog_rew={rew_d.item():+.4f}  "
                      f"arm_rew={rew_a.item():+.4f}  "
                      f"h={body_h:.3f}  resets={resets}  {fps_cur:.1f}fps")

            if done.any():
                resets += 1
                body_h = env.root_states[0, 2].item()
                print(f"  [Reset #{resets} at step {step+1}, h={body_h:.3f}]")
                env.commands_dog[:, 0] = 0.0
                env.commands_dog[:, 1:5] = 0.0
                env.dog_obs_history[:] = (
                    env.get_dog_observations()["obs"].repeat(1, dog_slots))

    elapsed = time.time() - t0
    print(f"\nDone: {args.num_steps} steps in {elapsed:.1f}s, "
          f"dog_rew={total_rew_dog:+.2f}  arm_rew={total_rew_arm:+.2f}  "
          f"resets={resets}")

    if frames:
        print(f"\nSaving {len(frames)} frames -> {args.video_path}")
        save_video(frames, args.video_path, args.fps)


if __name__ == "__main__":
    main()
