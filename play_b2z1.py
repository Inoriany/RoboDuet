"""
B2Z1 Play/Eval Script — Replicates EXACT config from auto_train.py
================================================================
Usage (run on remote server):
    python play_b2z1.py                          # 600 steps with viewer (needs X11)
    python play_b2z1.py --smoke_test             # 5 steps quick sanity check
    python play_b2z1.py --save_video             # headless + save .mp4 file
    python play_b2z1.py --save_video --num_steps 1000
    python play_b2z1.py --headless               # headless, no video
"""

# ============================================================
# CRITICAL: isaacgym MUST be imported before torch
# ============================================================
import isaacgym
assert isaacgym
import torch
import numpy as np
import sys
import os
import argparse
import time

# Ensure RoboDuet is on the path
sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym.utils.global_switch import global_switch

CHECKPOINT_PATH = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_dog/ac_weights_last_dog.pt"
)


def setup_config():
    """Replicate the EXACT config chain from auto_train.py.

    Order matters — config_go1 -> config_wtw -> config_asset -> overrides.
    Every override below is copied verbatim from auto_train.py so that the
    environment and observation/action dimensions match the trained checkpoint.
    """

    # --- Step 1: base config functions (same order as training) ---
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    # --- Step 2: all overrides from auto_train.py (exact order) ----------

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

    # *** OVERRIDE num_envs = 1 for inference ***
    Cfg.env.num_envs = 1

    Cfg.env.keep_arm_fixed = True

    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False

    Cfg.control.update_obs_freq = 20   # Hz

    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63

    # Reward scales (env init reads these even during inference)
    Cfg.hybrid.reward_scales.tracking_lin_vel = (
        0.7 * Cfg.reward_scales.tracking_lin_vel
    )
    Cfg.hybrid.reward_scales.tracking_ang_vel = (
        0.5 * Cfg.reward_scales.tracking_ang_vel
    )
    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy = -0.00004

    Cfg.reward_scales.jump = -0.00
    Cfg.rewards.terminal_body_height = 0.05   # extremely low — almost never triggers
    Cfg.rewards.use_terminal_body_height = False  # disable fall-reset entirely
    Cfg.env.max_episode_length = 99999   # no episode timeout

    # Disable domain randomization for inference — prevents re-randomization
    # on reset from invalidating the warm observation history.
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.push_robots = False

    # Force / perturbation
    Cfg.commands.T_force_range = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres = 0.3
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01

    # Observation flags
    Cfg.env.priv_observe_vel = False
    Cfg.commands.global_reference = False
    Cfg.env.priv_observe_high_freq_goal = False

    # Privileged obs counts
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9

    # Rendering / vision
    Cfg.asset.render_sphere = True
    Cfg.hybrid.use_vision = False

    # Manipulation reward weights
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1

    # Arm reward scales (derived from base scales)
    Cfg.hybrid.reward_scales.arm_dof_vel = (
        10 * Cfg.reward_scales.dof_vel
    )
    Cfg.hybrid.reward_scales.arm_dof_acc = (
        10 * Cfg.reward_scales.dof_acc
    )
    Cfg.hybrid.reward_scales.arm_action_rate = (
        10 * Cfg.reward_scales.action_rate
    )
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1
    )
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2
    )

    # Rotation representation (default: Euler, not rot6d)
    Cfg.use_rot6d = False

    # --- Step 3: B2Z1 robot asset ---
    Cfg.asset.file = (
        '{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf'
    )

    # --- Step 4: global switch — arm stays disabled (dog-only checkpoint) ---
    global_switch.pretrained_to_hybrid_start = 999999
    global_switch.pretrained_to_hybrid_end = 999999


def create_recording_camera(gym, sim, envs, width=1280, height=720):
    """Create an off-screen camera for programmatic video capture."""
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = False

    camera_handle = gym.create_camera_sensor(envs[0], camera_props)

    # Close front-quarter view: slightly in front/side, low angle
    cam_pos = gymapi.Vec3(1.0, 0.7, 0.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.25)
    gym.set_camera_location(camera_handle, envs[0], cam_pos, cam_target)

    return camera_handle, camera_props


def update_camera_to_follow_robot(gym, envs, camera_handle, root_states):
    """Update camera to follow the robot's base position."""
    rx = root_states[0, 0].item()
    ry = root_states[0, 1].item()
    rz = root_states[0, 2].item()

    # Offset: behind-right and slightly above the robot
    cam_pos = gymapi.Vec3(rx + 1.0, ry + 0.7, rz + 0.3)
    cam_target = gymapi.Vec3(rx, ry, rz)
    gym.set_camera_location(camera_handle, envs[0], cam_pos, cam_target)


def capture_frame(gym, sim, envs, camera_handle, camera_props,
                  root_states=None):
    """Capture a single RGBA frame from the camera."""
    # Update camera to track robot if root_states provided
    if root_states is not None:
        update_camera_to_follow_robot(gym, envs, camera_handle, root_states)

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    image = gym.get_camera_image(
        sim, envs[0], camera_handle, gymapi.IMAGE_COLOR
    )
    image = image.reshape(camera_props.height, camera_props.width, 4)
    return image[:, :, :3]  # drop alpha -> RGB


def save_video_from_frames(frames, output_path, fps=30):
    """Save list of RGB numpy arrays as mp4 video."""
    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                     quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[OK]  Video saved: {output_path}  "
              f"({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return True
    except ImportError:
        pass

    # Fallback: try cv2
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[OK]  Video saved (cv2): {output_path}  "
              f"({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return True
    except ImportError:
        pass

    # Last resort: save frames as images
    frames_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frames_dir, exist_ok=True)
    import struct, zlib
    for i, frame in enumerate(frames):
        # Save as simple PPM (no dependencies needed)
        ppm_path = os.path.join(frames_dir, f"frame_{i:04d}.ppm")
        h, w = frame.shape[:2]
        with open(ppm_path, 'wb') as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(frame.tobytes())
    print(f"[OK]  Saved {len(frames)} frames to {frames_dir}/")
    print("      Convert with: ffmpeg -r 30 -i frame_%04d.ppm -c:v libx264 out.mp4")
    return True


def main():
    parser = argparse.ArgumentParser(description="B2Z1 Play / Eval")
    parser.add_argument(
        "--num_steps", type=int, default=600,
        help="Number of simulation steps (default: 600)"
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run only 5 steps as a quick sanity check"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without the IsaacGym viewer"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Capture frames and save as mp4 (implies --headless)"
    )
    parser.add_argument(
        "--video_path", type=str, default="/root/RoboDuet/b2z1_play.mp4",
        help="Output video path (default: /root/RoboDuet/b2z1_play.mp4)"
    )
    parser.add_argument(
        "--video_fps", type=int, default=30,
        help="Video FPS (default: 30)"
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.num_steps = 5
    if args.save_video:
        args.headless = True  # video capture works in headless

    print("=" * 60)
    print("  B2Z1 Play/Eval — exact auto_train.py config replica")
    print(f"  Steps: {args.num_steps}  |  Headless: {args.headless}"
          f"  |  Save video: {args.save_video}")
    print("=" * 60)

    # Warn if DISPLAY is missing (X11 required for viewer)
    if not args.headless:
        display = os.environ.get("DISPLAY", "")
        if not display:
            print("[WARN] DISPLAY is not set — viewer may not appear.")
            print("       Use --headless or --save_video instead.")
        else:
            print(f"[OK] DISPLAY = {display}")

    # ---- Config --------------------------------------------------------
    setup_config()
    print(f"\n[Config]")
    print(f"  num_envs             = {Cfg.env.num_envs}")
    print(f"  num_actions          = {Cfg.env.num_actions}")
    print(f"  num_observations     = {Cfg.env.num_observations}")
    print(f"  dog_num_observations = {Cfg.dog.dog_num_observations}")
    print(f"  dog_num_obs_history  = {Cfg.dog.dog_num_obs_history}")
    print(f"  dog_num_priv_obs     = {Cfg.dog.dog_num_privileged_obs}")
    print(f"  keep_arm_fixed       = {Cfg.env.keep_arm_fixed}")
    print(f"  use_rot6d            = {Cfg.use_rot6d}")

    # ---- Environment ---------------------------------------------------
    print("\n[1/4] Creating environment ...")
    env = VelocityTrackingEasyEnv(
        sim_device="cuda:0",
        headless=args.headless,
        num_envs=1,          # also passed explicitly for safety
        cfg=Cfg,
    )
    env = HistoryWrapper(env)
    print("[OK]  Environment + HistoryWrapper ready")

    # ---- Video camera (if saving) --------------------------------------
    camera_handle = None
    camera_props = None
    frames = []
    if args.save_video:
        print("\n[+] Setting up recording camera ...")
        camera_handle, camera_props = create_recording_camera(
            env.gym, env.sim, env.envs
        )
        print(f"[OK]  Camera ready ({camera_props.width}x{camera_props.height})")

    # ---- Dog Policy ----------------------------------------------------
    print("\n[2/4] Loading dog policy ...")
    dog_obs_hist_dim = Cfg.dog.dog_num_obs_history    # 1680
    dog_priv_obs_dim = Cfg.dog.dog_num_privileged_obs  # 2
    dog_action_dim = 12                                # 12 leg joints

    policy = DogActorCritic(
        num_obs=1,                           # placeholder (unused directly)
        num_privileged_obs=dog_priv_obs_dim,
        num_obs_history=dog_obs_hist_dim,
        num_actions=dog_action_dim,
    ).to("cuda:0")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cuda:0")
    policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[OK]  Loaded checkpoint  (obs_hist={dog_obs_hist_dim}, "
          f"priv={dog_priv_obs_dim}, act={dog_action_dim})")

    # ---- Reset ---------------------------------------------------------
    print("\n[3/4] Resetting environment ...")
    env.reset()

    # Arm action dim: total actions - dog actions
    num_arm_actions = Cfg.env.num_actions - dog_action_dim  # 18 - 12 = 6

    # ---- Set gentle velocity commands ----------------------------------
    env.commands_dog[:, 0] = 0.0   # lin_vel_x  = 0.0 (standing)
    env.commands_dog[:, 1] = 0.0   # lin_vel_y  = 0
    env.commands_dog[:, 2] = 0.0   # ang_vel    = 0 (no turning)
    env.commands_dog[:, 3] = 0.0   # body pitch = 0
    env.commands_dog[:, 4] = 0.0   # body roll  = 0
    print("[OK]  Environment reset  (commands: vx=0.0, teacher mode, no terminal)")

    # ---- Pre-fill obs_history with first real observation --------------
    dog_obs_first = env.get_dog_observations()
    first_obs = dog_obs_first["obs"]
    dog_num_obs = Cfg.dog.dog_num_observations
    num_history_slots = Cfg.dog.dog_num_obs_history // dog_num_obs
    env.dog_obs_history[:] = first_obs.repeat(1, num_history_slots)
    print(f"[OK]  Pre-filled dog_obs_history  ({num_history_slots} slots x {dog_num_obs} obs)")

    # ---- Warmup phase (teacher mode, short) ----------------------------
    WARMUP_STEPS = 50
    print(f"\n[3.5/4] Warming up obs_history ({WARMUP_STEPS} teacher-mode steps) ...")

    with torch.no_grad():
        for ws in range(WARMUP_STEPS):
            dog_obs = env.get_dog_observations()
            obs_history = dog_obs["obs_history"]
            priv_obs = dog_obs["privileged_obs"]     # [1, 2] ground truth
            action_dog = policy.act_teacher(obs_history, priv_obs)
            action_arm = torch.zeros(1, num_arm_actions, device="cuda:0")
            rew_dog, rew_arm, done, info = env.step(action_dog, action_arm)
            env.commands_dog[:, 0] = 0.0
            env.commands_dog[:, 1:5] = 0.0
            if done.any():
                print(f"    [Warmup reset at step {ws+1}]")
                dog_obs_w = env.get_dog_observations()
                env.dog_obs_history[:] = dog_obs_w["obs"].repeat(1, num_history_slots)

    print("[OK]  Warmup complete\n")

    # ---- Inference Loop (TEACHER MODE — uses real privileged obs) ------
    print(f"[4/4] Running {args.num_steps} inference steps (TEACHER MODE) ...")
    if not args.headless:
        print("=" * 60)
        print("  >>> IsaacGym viewer should now be visible <<<")
        print("  >>> Record screen with OBS / Xbox Game Bar <<<")
        print("=" * 60)
    if args.save_video:
        print(f"  Capturing frames -> {args.video_path}")
    print()

    t0 = time.time()
    total_reward = 0.0
    num_resets = 0
    step = -1

    try:
        with torch.no_grad():
            for step in range(args.num_steps):
                # 1. Get dog observation dict
                dog_obs = env.get_dog_observations()
                obs_history = dog_obs["obs_history"]      # [1, 1680]
                priv_obs = dog_obs["privileged_obs"]      # [1, 2] ground truth

                # 2. TEACHER policy forward (uses real privileged obs)
                action_dog = policy.act_teacher(obs_history, priv_obs)  # [1, 12]

                # 3. Arm = zeros (keep_arm_fixed, switch_open=False)
                action_arm = torch.zeros(
                    1, num_arm_actions, device="cuda:0"
                )

                # 4. Step the wrapped environment
                rew_dog, rew_arm, done, info = env.step(
                    action_dog, action_arm
                )

                total_reward += rew_dog.item()

                # Keep commands: standing still
                env.commands_dog[:, 0] = 0.0   # vx = 0.0
                env.commands_dog[:, 1:5] = 0.0  # vy, yaw, pitch, roll = 0

                # 5. Capture frame for video (camera follows robot)
                if args.save_video and camera_handle is not None:
                    root_st = env.root_states[:1]  # [1, 13]
                    frame = capture_frame(
                        env.gym, env.sim, env.envs,
                        camera_handle, camera_props,
                        root_states=root_st
                    )
                    frames.append(frame)

                # Progress + print body height for debugging
                if step == 0 or (step + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    fps = (step + 1) / max(elapsed, 1e-6)
                    body_h = env.root_states[0, 2].item()
                    print(
                        f"  Step {step+1:>4d}/{args.num_steps}  "
                        f"| dog_rew {rew_dog.item():+.4f}  "
                        f"| cum_rew {total_reward:+.2f}  "
                        f"| h={body_h:.3f}  "
                        f"| resets {num_resets}"
                        f"  | {fps:.1f} fps"
                    )

                # Auto-reset on terminal
                if done.any():
                    num_resets += 1
                    if num_resets <= 10:
                        body_h = env.root_states[0, 2].item()
                        print(f"  [Reset #{num_resets} at step {step+1}, h={body_h:.3f}]")
                    env.commands_dog[:, 0] = 0.0   # restore vx=0
                    env.commands_dog[:, 1:5] = 0.0
                    # Re-fill history after reset
                    dog_obs_r = env.get_dog_observations()
                    env.dog_obs_history[:] = dog_obs_r["obs"].repeat(1, num_history_slots)

    except KeyboardInterrupt:
        print(f"\n  [Interrupted at step {step+1}]")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Finished {args.num_steps} steps in {elapsed:.1f}s")
    print(f"  Cumulative dog reward: {total_reward:+.2f}")

    # ---- Save video ----------------------------------------------------
    if args.save_video and len(frames) > 0:
        print(f"\n  Saving {len(frames)} frames as video ...")
        save_video_from_frames(frames, args.video_path, fps=args.video_fps)

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
