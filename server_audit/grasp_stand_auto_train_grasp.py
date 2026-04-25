
import isaacgym
assert isaacgym
import torch
import argparse

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset

import wandb
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from go1_gym import MINI_GYM_ROOT_DIR
import shutil
import pickle

from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic import Runner
from go1_gym_learn.ppo_cse_automatic.ppo import PPO_Args
from go1_gym_learn.ppo_cse_automatic import RunnerArgs, ArmRunnerArgs, DogRunnerArgs
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC_Args
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmAC_Args
from go1_gym.utils import format_code, set_seed, global_switch

os.environ["WANDB_SILENT"] = "true"


def train_grasp(args):

    if args.debug:
        mode = "disabled"
        args.num_envs = 12
    else:
        mode = "online"
        if args.offline:
            mode = "offline"

    if args.no_wandb:
        mode = "disabled"

    args.seed = set_seed(args.seed)
    args.tags.append(f"seed{args.seed}")

    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    # ================================================================
    # FIX 1: Correct PD gains (same as locomotion training)
    # ================================================================
    KP_LEG = 200.0
    KD_LEG = 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP_LEG
    Cfg.dog.control.damping_leg["joint"]   = KD_LEG
    Cfg.control.stiffness["joint"]         = KP_LEG
    Cfg.control.damping["joint"]           = KD_LEG
    print(f"[KP/KD] stiffness_leg={KP_LEG}, damping_leg={KD_LEG}", flush=True)

    # ================================================================
    # FIX 2: ARM ENABLED from step 0 - no two-stage
    # ================================================================
    Cfg.env.keep_arm_fixed = False   # arm can move freely
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0
    print("[ARM] keep_arm_fixed=False, arm training starts at iteration 0", flush=True)

    # ================================================================
    # FIX 3: ZERO locomotion commands -> dog stands still
    # This forces the dog to learn balance while arm manipulates
    # ================================================================
    Cfg.commands.lin_vel_x    = [0.0, 0.0]
    Cfg.commands.lin_vel_y    = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw  = [0.0, 0.0]
    Cfg.commands.limit_vel_x  = [0.0, 0.0]
    Cfg.commands.limit_vel_y  = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]
    print("[CMD] locomotion velocity = 0 (stand still + arm manipulation)", flush=True)

    # ================================================================
    # Standard config (same as auto_train.py)
    # ================================================================
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

    Cfg.env.num_envs = args.num_envs
    Cfg.terrain.mesh_type = "plane"
    if Cfg.terrain.mesh_type == "plane":
        Cfg.terrain.teleport_robots = False
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions      = 18
    Cfg.env.num_observations = 63

    # Locomotion rewards (reduced - robot just needs to stand)
    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0   # no loco reward needed
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.reward_scales.loco_energy        = -0.00004
    Cfg.reward_scales.jump = -0.00
    Cfg.rewards.terminal_body_height     = 0.28
    Cfg.rewards.use_terminal_body_height = True

    # Arm manipulation rewards (boosted)
    Cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 5.0   # default 1.0 -> 5.0
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1

    DogRunnerArgs.resume = False
    ArmRunnerArgs.resume = False

    if args.debug:
        RunnerArgs.save_interval = 2
        RunnerArgs.save_video_interval = 10

    Cfg.commands.T_force_range = [2, 4.]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres  = 0.3
    Cfg.domain_rand.max_force     = 15
    Cfg.domain_rand.max_force_offset = 0.01
    Cfg.env.priv_observe_vel          = False
    Cfg.commands.global_reference     = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs    = 2
    Cfg.arm.arm_num_privileged_obs    = 9
    Cfg.env.num_privileged_obs        = 9
    Cfg.asset.render_sphere           = True
    Cfg.hybrid.use_vision             = False
    Cfg.hybrid.reward_scales.arm_dof_vel = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * Cfg.reward_scales.action_smoothness_1
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * Cfg.reward_scales.action_smoothness_2
    Cfg.use_rot6d = False

    global_switch.init_sigmoid_lr()

    # B2Z1 robot
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"

    if args.headless:
        RunnerArgs.log_video = False

    now = datetime.now()
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

    args.log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}", wandb.run.name)
    args.log_dir += f"_seed{args.seed}"

    if not args.debug:
        os.makedirs(osp.join(args.log_dir, "checkpoints_arm"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "checkpoints_dog"), exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "scripts"),         exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "videos"),          exist_ok=True)
        os.makedirs(osp.join(args.log_dir, "deploy_model"),    exist_ok=True)
        os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model",   exist_ok=True)

        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/scripts/auto_train_grasp.py",
            f"{args.log_dir}/scripts/auto_train_grasp.py",
        )
        shutil.copyfile(
            f"{MINI_GYM_ROOT_DIR}/go1_gym/envs/automatic/legged_robot.py",
            f"{args.log_dir}/scripts/legged_robot.py",
        )
        temp_dict = {"Cfg": vars(Cfg), "KP_LEG": KP_LEG, "KD_LEG": KD_LEG}
        with open(f"{args.log_dir}/params.txt", "w", encoding="utf-8") as f:
            f.write(str(temp_dict))
        with open(osp.join(args.log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump(temp_dict, f)

    env = VelocityTrackingEasyEnv(
        sim_device=args.sim_device, headless=args.headless, cfg=Cfg
    )
    env = HistoryWrapper(env)
    gpu_id = args.sim_device.split(":")[-1]
    runner = Runner(
        env,
        device=f"cuda:{gpu_id}",
        run_name=args.run_name,
        resume=False,
        log_dir=args.log_dir,
        debug=args.debug,
    )
    runner.learn(
        num_learning_iterations=args.num_learning_iterations,
        init_at_random_ep_len=True,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B2Z1 Grasping Training - Stand Still + Arm Manipulation")
    parser.add_argument("--headless",                 action="store_true", default=False)
    parser.add_argument("--sim_device",               type=str,  default="cuda:0")
    parser.add_argument("--num_learning_iterations",  type=int,  default=30000)
    parser.add_argument("--eval_freq",                type=int,  default=100)
    parser.add_argument("--num_envs",                 type=int,  default=2048)
    parser.add_argument("--run_name",                 type=str,  default="b2z1_grasp_stand")
    parser.add_argument("--debug",                    action="store_true")
    parser.add_argument("--offline",                  action="store_true")
    parser.add_argument("--no_wandb",                 action="store_true")
    parser.add_argument("--tags",  nargs="+", default=[])
    parser.add_argument("--notes", type=str,  default=None)
    parser.add_argument("--seed",  type=int,  default=-1)
    args = parser.parse_args()
    train_grasp(args)
