"""
auto_train_hybrid.py -- Hot-start Walk + Grasp Hybrid Training
==============================================================
Loads the best dog checkpoint from b2z1_kp200_kd20 (locomotion)
and the best arm checkpoint from b2z1_grasp_stand (manipulation),
then trains them together in a combined walk-while-grasping task.

Usage (on server, inside conda roboduet):
    python /root/RoboDuet/scripts/auto_train_hybrid.py \\
        --headless --no_wandb \\
        --run_name b2z1_hybrid \\
        --num_learning_iterations 30000 \\
        --num_envs 2048

    # Override checkpoint paths if needed:
    python /root/RoboDuet/scripts/auto_train_hybrid.py \\
        --headless --no_wandb \\
        --dog_ckpt /path/to/ac_weights_last_dog.pt \\
        --arm_ckpt /path/to/ac_weights_last_arm.pt
"""

# ============================================================
# CRITICAL: isaacgym MUST come before torch
# ============================================================
import isaacgym
assert isaacgym
import torch
import argparse
import glob
import os
import os.path as osp
import shutil
import pickle
from datetime import datetime
from pathlib import Path

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset

import wandb
from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic import Runner
from go1_gym_learn.ppo_cse_automatic import RunnerArgs, ArmRunnerArgs, DogRunnerArgs
from go1_gym.utils import set_seed, global_switch

os.environ["WANDB_SILENT"] = "true"

RUNS_ROOT = f"{MINI_GYM_ROOT_DIR}/runs"


# ============================================================
# Helpers: auto-detect latest checkpoints
# ============================================================
def find_latest_dog_ckpt(run_name="b2z1_kp200_kd20"):
    """Find the most recently modified dog checkpoint in a run."""
    pattern = osp.join(RUNS_ROOT, run_name, "**",
                       "checkpoints_dog", "ac_weights_last_dog.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        pattern2 = osp.join(RUNS_ROOT, run_name, "**",
                            "checkpoints_dog", "ac_weights_*.pt")
        candidates = glob.glob(pattern2, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No dog checkpoint found under {RUNS_ROOT}/{run_name}"
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def find_latest_arm_ckpt(run_name="b2z1_grasp_stand"):
    """Find the most recently modified arm checkpoint in a run."""
    pattern = osp.join(RUNS_ROOT, run_name, "**",
                       "checkpoints_arm", "ac_weights_last_arm.pt")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        pattern2 = osp.join(RUNS_ROOT, run_name, "**",
                            "checkpoints_arm", "ac_weights_*.pt")
        candidates = glob.glob(pattern2, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No arm checkpoint found under {RUNS_ROOT}/{run_name}"
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


# ============================================================
# Training function
# ============================================================
def train_hybrid(args):

    mode = "disabled" if (args.no_wandb or args.debug) else "online"

    args.seed = set_seed(args.seed)
    args.tags.append(f"seed{args.seed}")

    # ---- Find checkpoints -----------------------------------------------
    dog_ckpt = args.dog_ckpt or find_latest_dog_ckpt(args.loco_run)
    arm_ckpt = args.arm_ckpt or find_latest_arm_ckpt(args.grasp_run)
    print(f"[DOG CKPT] {dog_ckpt}", flush=True)
    print(f"[ARM CKPT] {arm_ckpt}", flush=True)

    # ---- Config chain (same as auto_train_kp200.py) -------------------
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    # Correct PD gains
    KP_LEG, KD_LEG = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP_LEG
    Cfg.dog.control.damping_leg["joint"]   = KD_LEG
    Cfg.control.stiffness["joint"]         = KP_LEG
    Cfg.control.damping["joint"]           = KD_LEG
    print(f"[KP/KD] {KP_LEG}/{KD_LEG}", flush=True)

    # ARM ENABLED -- hybrid mode
    Cfg.env.keep_arm_fixed = False
    # Both policies resume from pre-trained checkpoints from step 0
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0
    print("[ARM] keep_arm_fixed=False, hybrid active from iteration 0", flush=True)

    # Walking velocity commands ENABLED (dog walks while arm manipulates)
    # Keep default command ranges -- locomotion curriculum will handle it
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

    Cfg.env.num_envs = args.num_envs if not args.debug else 12
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions      = 18
    Cfg.env.num_observations = 63

    # Reward scales: keep locomotion rewards + arm manipulation rewards
    Cfg.hybrid.reward_scales.tracking_lin_vel = (
        0.7 * Cfg.reward_scales.tracking_lin_vel)
    Cfg.hybrid.reward_scales.tracking_ang_vel = (
        0.5 * Cfg.reward_scales.tracking_ang_vel)
    Cfg.hybrid.reward_scales.arm_energy   = -0.00004
    Cfg.reward_scales.loco_energy         = -0.00004
    Cfg.reward_scales.jump                = -0.00
    Cfg.rewards.terminal_body_height      = 0.28
    Cfg.rewards.use_terminal_body_height  = True

    # Arm manipulation rewards
    Cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 3.0
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1

    # Resume from pre-trained checkpoints (HOT START)
    DogRunnerArgs.resume      = True
    DogRunnerArgs.resume_path = dog_ckpt
    ArmRunnerArgs.resume      = True
    ArmRunnerArgs.resume_path = arm_ckpt
    print(f"[RESUME] Dog: {dog_ckpt}", flush=True)
    print(f"[RESUME] Arm: {arm_ckpt}", flush=True)

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
    Cfg.hybrid.reward_scales.arm_dof_vel   = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc   = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = (
        10 * Cfg.reward_scales.action_rate)
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1)
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2)
    Cfg.use_rot6d = False

    global_switch.init_sigmoid_lr()

    Cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    )

    if args.headless:
        RunnerArgs.log_video = False

    if args.debug:
        RunnerArgs.save_interval      = 2
        RunnerArgs.save_video_interval = 10

    # ---- WandB + log dir -----------------------------------------------
    now  = datetime.now()
    stem = Path(__file__).stem
    wandb.init(
        entity="yuxiaozhao-the-chinese-university-of-hong-kong",
        project="dev",
        group=args.run_name,
        mode=mode,
        notes=args.notes,
        name=f'{now.strftime("%Y-%m-%d")}/{stem}/{now.strftime("%H%M%S.%f")}',
        tags=args.tags,
        dir=f"{MINI_GYM_ROOT_DIR}",
    )

    args.log_dir  = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}",
                             wandb.run.name)
    args.log_dir += f"_seed{args.seed}"

    if not args.debug:
        for sub in ("checkpoints_arm", "checkpoints_dog", "scripts",
                    "videos", "deploy_model"):
            os.makedirs(osp.join(args.log_dir, sub), exist_ok=True)
        os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

        # Save this script alongside the run for reproducibility
        shutil.copyfile(__file__,
                        osp.join(args.log_dir, "scripts", "auto_train_hybrid.py"))

        params = {
            "Cfg": vars(Cfg),
            "KP_LEG": KP_LEG, "KD_LEG": KD_LEG,
            "dog_ckpt": dog_ckpt, "arm_ckpt": arm_ckpt,
        }
        with open(osp.join(args.log_dir, "params.txt"), "w") as f:
            f.write(str(params))
        with open(osp.join(args.log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump(params, f)

    # ---- Environment ---------------------------------------------------
    env = VelocityTrackingEasyEnv(
        sim_device=args.sim_device, headless=args.headless, cfg=Cfg
    )
    env = HistoryWrapper(env)

    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(
        env,
        device=f"cuda:{gpu_id}",
        run_name=args.run_name,
        resume=True,      # Runner itself also needs resume=True
        log_dir=args.log_dir,
        debug=args.debug,
    )
    runner.learn(
        num_learning_iterations=args.num_learning_iterations,
        init_at_random_ep_len=True,
        eval_freq=args.eval_freq,
    )


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B2Z1 Hybrid Training: Walk + Grasp (hot-start)"
    )
    parser.add_argument("--headless",  action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_learning_iterations", type=int, default=30000)
    parser.add_argument("--eval_freq",  type=int,  default=100)
    parser.add_argument("--num_envs",   type=int,  default=2048)
    parser.add_argument("--run_name",   type=str,  default="b2z1_hybrid")
    # Hot-start source runs (auto-detected unless overridden)
    parser.add_argument("--loco_run",  type=str,  default="b2z1_kp200_kd20",
                        help="Run name to pull dog checkpoint from")
    parser.add_argument("--grasp_run", type=str,  default="b2z1_grasp_stand",
                        help="Run name to pull arm checkpoint from")
    # Manual checkpoint overrides (optional)
    parser.add_argument("--dog_ckpt",  type=str,  default=None,
                        help="Explicit path to dog .pt checkpoint")
    parser.add_argument("--arm_ckpt",  type=str,  default=None,
                        help="Explicit path to arm .pt checkpoint")
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--no_wandb",  action="store_true")
    parser.add_argument("--offline",   action="store_true")
    parser.add_argument("--tags",  nargs="+", default=[])
    parser.add_argument("--notes", type=str,  default=None)
    parser.add_argument("--seed",  type=int,  default=-1)
    args = parser.parse_args()
    train_hybrid(args)
