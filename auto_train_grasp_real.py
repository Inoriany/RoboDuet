import isaacgym
assert isaacgym
import torch
import argparse
import os
import os.path as osp
import shutil
import pickle
from datetime import datetime
from pathlib import Path

import wandb

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper
from go1_gym_learn.ppo_cse_automatic import Runner
from go1_gym_learn.ppo_cse_automatic import RunnerArgs, ArmRunnerArgs, DogRunnerArgs
from go1_gym.utils import set_seed, global_switch

from real_grasp_env import RealGraspEnv


os.environ["WANDB_SILENT"] = "true"


def add_real_grasp_cfg():
    class RealGraspCfg:
        box_size = 0.05
        spawn_x = [0.40, 0.58]
        spawn_y = [-0.08, 0.08]
        spawn_z = [0.52, 0.68]
        stage1_spawn_x = [0.44, 0.50]
        stage1_spawn_y = [-0.03, 0.03]
        stage1_spawn_z = [0.58, 0.64]
        stage2_spawn_x = [0.42, 0.54]
        stage2_spawn_y = [-0.05, 0.05]
        stage2_spawn_z = [0.55, 0.66]
        curr_stage1_steps = 150000
        curr_stage2_steps = 500000
        close_thresh = 0.08
        fix_object = False
        disable_object_gravity = False
        script_gripper = True
        auto_close_dist = 0.07
        open_gripper = 0.0
        closed_gripper = -0.8
    Cfg.real_grasp = RealGraspCfg


def train(args):
    mode = "disabled" if (args.no_wandb or args.debug) else ("offline" if args.offline else "online")
    args.seed = set_seed(args.seed)

    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)
    add_real_grasp_cfg()

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
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01
    Cfg.control.control_type = "M"
    Cfg.env.observe_two_prev_actions = False
    Cfg.commands.body_roll_range = [-0.4, 0.4]
    Cfg.commands.limit_body_roll = [-0.4, 0.4]
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]

    Cfg.env.num_envs = 12 if args.debug else args.num_envs
    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63
    Cfg.env.priv_observe_vel = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.commands.global_reference = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9

    Cfg.hybrid.use_vision = True
    Cfg.rewards.reward_container_name = "Rewards"
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1
    Cfg.hybrid.rewards.use_terminal_roll = False
    Cfg.hybrid.rewards.use_terminal_pitch = False
    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_manip_commands_tracking_combine = 0.0
    Cfg.hybrid.reward_scales.vis_manip_commands_tracking_lpy = 0.0
    Cfg.hybrid.reward_scales.vis_manip_commands_tracking_rpy = 0.0
    Cfg.hybrid.reward_scales.orientation_heuristic = 0.0
    Cfg.hybrid.reward_scales.orientation_control = -2.0
    Cfg.hybrid.reward_scales.hip_action_l2 = -0.02
    Cfg.hybrid.reward_scales.arm_energy = -0.00004
    Cfg.hybrid.reward_scales.arm_dof_vel = -0.001
    Cfg.hybrid.reward_scales.arm_dof_acc = -2.5e-6
    Cfg.hybrid.reward_scales.arm_action_rate = -0.05
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = -0.2
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = -0.2

    # Placeholder: these names will require reward-container integration on server.
    Cfg.hybrid.reward_scales.grasp_obj_dist = 6.0
    Cfg.hybrid.reward_scales.grasp_xy_align = 4.0
    Cfg.hybrid.reward_scales.grasp_z_align = 2.0
    Cfg.hybrid.reward_scales.grasp_close_bonus = 3.0
    Cfg.hybrid.reward_scales.grasp_lift_height = 6.0
    Cfg.hybrid.reward_scales.grasp_drop_penalty = -4.0

    Cfg.rewards.terminal_body_height = 0.30
    Cfg.rewards.use_terminal_body_height = True
    Cfg.asset.render_sphere = True
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"

    global_switch.init_sigmoid_lr()
    global_switch.open_switch()

    DogRunnerArgs.resume = args.resume
    ArmRunnerArgs.resume = args.resume
    DogRunnerArgs.resume_path = args.dog_resume_path
    ArmRunnerArgs.resume_path = args.arm_resume_path

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
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "checkpoints_arm"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "checkpoints_dog"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "scripts"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "videos"), exist_ok=True)
    os.makedirs(osp.join(args.log_dir, "deploy_model"), exist_ok=True)
    os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

    if not args.debug:
        shutil.copyfile(__file__, osp.join(args.log_dir, "scripts", "auto_train_grasp_real.py"))
        for extra in ["real_grasp_env.py", "real_grasp_rewards.py"]:
            src = osp.join(MINI_GYM_ROOT_DIR, extra)
            if osp.exists(src):
                shutil.copyfile(src, osp.join(args.log_dir, "scripts", extra))
        with open(osp.join(args.log_dir, "params.txt"), "w", encoding="utf-8") as f:
            f.write(str({"Cfg": vars(Cfg), "real_grasp": vars(Cfg.real_grasp)}))
        with open(osp.join(args.log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump({"note": "real_grasp_cfg omitted from pickle; see params.txt"}, f)

    env = RealGraspEnv(sim_device=args.sim_device, headless=args.headless, cfg=Cfg)
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
    parser = argparse.ArgumentParser(description="Real-object B2Z1 grasp training scaffold")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_learning_iterations", type=int, default=30000)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--run_name", type=str, default="b2z1_grasp_real")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm_resume_path", type=str, default="")
    parser.add_argument("--dog_resume_path", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    train(parser.parse_args())
