"""
test_walk.py -- Video generation for kp200 walking checkpoint
=============================================================
Auto-finds the latest dog checkpoint from b2z1_kp200_kd20 run,
runs walking inference, and saves an .mp4 video.

Usage (on server, inside conda roboduet):
    python /root/RoboDuet/test_walk.py
    python /root/RoboDuet/test_walk.py --vx 0.5 --num_steps 600
    python /root/RoboDuet/test_walk.py --checkpoint /path/to/ac_weights_last_dog.pt
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
from go1_gym.utils.global_switch import global_switch


# ============================================================
# Helper: find latest dog checkpoint from b2z1_kp200_kd20
# ============================================================
def find_latest_checkpoint(run_name="b2z1_kp200_kd20",
                            runs_root="/root/RoboDuet/runs"):
    """Return path to the most recently modified dog checkpoint in the run."""
    pattern = os.path.join(runs_root, run_name, "**",
                           "checkpoints_dog", "ac_weights_last_dog.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        # Fallback: any numbered checkpoint
        pattern2 = os.path.join(runs_root, run_name, "**",
                                "checkpoints_dog", "ac_weights_*.pt")
        candidates = glob.glob(pattern2, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No dog checkpoints found under {runs_root}/{run_name}. "
            "Is the training finished?"
        )
    # Pick the most recently modified file
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


# ============================================================
# Config -- mirrors auto_train_kp200.py EXACTLY
# ============================================================
def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    # Correct PD gains (same fix as auto_train_kp200.py)
    KP_LEG, KD_LEG = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP_LEG
    Cfg.dog.control.damping_leg["joint"]   = KD_LEG
    Cfg.control.stiffness["joint"]         = KP_LEG
    Cfg.control.damping["joint"]           = KD_LEG

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

    Cfg.env.num_envs = 1          # single env for inference
    Cfg.env.keep_arm_fixed = True
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions      = 18
    Cfg.env.num_observations = 63

    Cfg.hybrid.reward_scales.tracking_lin_vel = (
        0.7 * Cfg.reward_scales.tracking_lin_vel)
    Cfg.hybrid.reward_scales.tracking_ang_vel = (
        0.5 * Cfg.reward_scales.tracking_ang_vel)
    Cfg.hybrid.reward_scales.arm_energy   = -0.00004
    Cfg.reward_scales.loco_energy         = -0.00004
    Cfg.reward_scales.jump                = -0.00
    Cfg.rewards.terminal_body_height      = 0.05   # very low -> almost never triggers
    Cfg.rewards.use_terminal_body_height  = False  # disable fall-reset for video

    # No domain randomisation during eval
    Cfg.domain_rand.randomize_friction        = False
    Cfg.domain_rand.randomize_base_mass       = False
    Cfg.domain_rand.randomize_restitution     = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength  = False
    Cfg.domain_rand.randomize_motor_offset    = False
    Cfg.domain_rand.randomize_gravity         = False
    Cfg.domain_rand.push_robots               = False

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
    # Arm disabled globally (dog-only checkpoint)
    global_switch.pretrained_to_hybrid_start = 999999
    global_switch.pretrained_to_hybrid_end   = 999999


# ============================================================
# Camera helpers (same as play_b2z1.py)
# ============================================================
def create_recording_camera(gym, sim, envs, width=1280, height=720):
    camera_props = gymapi.CameraProperties()
    camera_props.width  = width
    camera_props.height = height
    camera_props.enable_tensors = False
    camera_handle = gym.create_camera_sensor(envs[0], camera_props)
    cam_pos    = gymapi.Vec3(1.0, 0.7, 0.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.25)
    gym.set_camera_location(camera_handle, envs[0], cam_pos, cam_target)
    return camera_handle, camera_props


def update_camera(gym, envs, camera_handle, root_states):
    rx = root_states[0, 0].item()
    ry = root_states[0, 1].item()
    rz = root_states[0, 2].item()
    cam_pos    = gymapi.Vec3(rx + 1.2, ry + 0.8, rz + 0.4)
    cam_target = gymapi.Vec3(rx, ry, rz)
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
    parser = argparse.ArgumentParser(description="Test walking (kp200) checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to dog checkpoint .pt (auto-detected if omitted)")
    parser.add_argument("--run_name", type=str, default="b2z1_kp200_kd20",
                        help="Run name to search for checkpoint")
    parser.add_argument("--vx", type=float, default=0.5,
                        help="Forward velocity command (default 0.5 m/s)")
    parser.add_argument("--num_steps", type=int, default=600,
                        help="Number of simulation steps (default 600 = 20s @ 30fps)")
    parser.add_argument("--video_path", type=str,
                        default="/root/RoboDuet/b2z1_walk_kp200.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_steps", type=int, default=50)
    args = parser.parse_args()

    # ---- Find checkpoint -----------------------------------------------
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        print(f"[Auto] Searching for latest checkpoint in {args.run_name} ...")
        ckpt_path = find_latest_checkpoint(args.run_name)
    print(f"[OK]  Checkpoint: {ckpt_path}")

    # ---- Config --------------------------------------------------------
    setup_config()
    print(f"\n[Config] num_actions={Cfg.env.num_actions}  "
          f"num_obs={Cfg.env.num_observations}  vx={args.vx}")

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

    # ---- Policy --------------------------------------------------------
    print("\n[2/4] Loading dog policy ...")
    dog_obs_hist_dim  = Cfg.dog.dog_num_obs_history
    dog_priv_obs_dim  = Cfg.dog.dog_num_privileged_obs
    dog_action_dim    = 12
    num_arm_actions   = Cfg.env.num_actions - dog_action_dim

    policy = DogActorCritic(
        num_obs=1,
        num_privileged_obs=dog_priv_obs_dim,
        num_obs_history=dog_obs_hist_dim,
        num_actions=dog_action_dim,
    ).to("cuda:0")

    ckpt = torch.load(ckpt_path, map_location="cuda:0")
    policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[OK]  Policy loaded  (hist={dog_obs_hist_dim}, priv={dog_priv_obs_dim})")

    # ---- Reset + set commands -----------------------------------------
    print("\n[3/4] Resetting environment ...")
    env.reset()
    env.commands_dog[:, 0] = args.vx    # forward velocity
    env.commands_dog[:, 1] = 0.0         # lateral velocity
    env.commands_dog[:, 2] = 0.0         # yaw rate
    env.commands_dog[:, 3] = 0.0         # body pitch
    env.commands_dog[:, 4] = 0.0         # body roll

    # Pre-fill obs history
    dog_num_obs = Cfg.dog.dog_num_observations
    num_slots   = dog_obs_hist_dim // dog_num_obs
    first_obs   = env.get_dog_observations()["obs"]
    env.dog_obs_history[:] = first_obs.repeat(1, num_slots)

    # ---- Warmup --------------------------------------------------------
    print(f"[3.5/4] Warmup ({args.warmup_steps} steps) ...")
    with torch.no_grad():
        for ws in range(args.warmup_steps):
            dog_obs  = env.get_dog_observations()
            action_d = policy.act_teacher(
                dog_obs["obs_history"], dog_obs["privileged_obs"])
            action_a = torch.zeros(1, num_arm_actions, device="cuda:0")
            _, _, done, _ = env.step(action_d, action_a)
            env.commands_dog[:, 0] = args.vx
            env.commands_dog[:, 1:5] = 0.0
            if done.any():
                env.dog_obs_history[:] = (
                    env.get_dog_observations()["obs"].repeat(1, num_slots))
    print("[OK]  Warmup done")

    # ---- Inference loop ------------------------------------------------
    print(f"\n[4/4] Running {args.num_steps} steps  (vx={args.vx}) ...")
    frames = []
    total_rew = 0.0
    resets = 0
    t0 = time.time()

    with torch.no_grad():
        for step in range(args.num_steps):
            dog_obs  = env.get_dog_observations()
            action_d = policy.act_teacher(
                dog_obs["obs_history"], dog_obs["privileged_obs"])
            action_a = torch.zeros(1, num_arm_actions, device="cuda:0")
            rew_d, _, done, _ = env.step(action_d, action_a)
            total_rew += rew_d.item()

            env.commands_dog[:, 0] = args.vx
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
                      f"rew={rew_d.item():+.4f}  h={body_h:.3f}  "
                      f"resets={resets}  {fps_cur:.1f}fps")

            if done.any():
                resets += 1
                body_h = env.root_states[0, 2].item()
                print(f"  [Reset #{resets} at step {step+1}, h={body_h:.3f}]")
                env.commands_dog[:, 0] = args.vx
                env.commands_dog[:, 1:5] = 0.0
                env.dog_obs_history[:] = (
                    env.get_dog_observations()["obs"].repeat(1, num_slots))

    elapsed = time.time() - t0
    print(f"\nDone: {args.num_steps} steps in {elapsed:.1f}s, "
          f"cumulative reward={total_rew:+.2f}, resets={resets}")

    # ---- Save video ----------------------------------------------------
    if frames:
        print(f"\nSaving {len(frames)} frames -> {args.video_path}")
        save_video(frames, args.video_path, args.fps)


if __name__ == "__main__":
    main()
