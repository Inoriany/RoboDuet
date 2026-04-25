import isaacgym
assert isaacgym
import argparse
import os
import os.path as osp
import shutil
import pickle
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import wandb

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper
from go1_gym.utils import set_seed, global_switch
from go1_gym_learn.ppo_cse_automatic import RunnerArgs, ArmRunnerArgs
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym_learn.ppo_cse_automatic.ppo import PPO, PPO_Args

from real_grasp_env import RealGraspEnv


os.environ["WANDB_SILENT"] = "true"


class RunningMeanStd:
    """Welford's online algorithm for running mean/std of rewards."""
    def __init__(self, device='cpu'):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var() + 1e-8
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return x / (self.var.sqrt().clamp(min=1e-4))


def add_cfg():
    class RealGraspCfg:
        box_size = 0.05
        spawn_x = [0.28, 0.40]
        spawn_y = [-0.015, 0.015]
        spawn_z = [0.580, 0.594]
        stage1_spawn_x = [0.32, 0.36]
        stage1_spawn_y = [-0.005, 0.005]
        stage1_spawn_z = [0.583, 0.591]
        stage2_spawn_x = [0.30, 0.38]
        stage2_spawn_y = [-0.010, 0.010]
        stage2_spawn_z = [0.581, 0.593]
        curr_stage1_steps = 20000000
        curr_stage2_steps = 20000000
        close_thresh = 0.50
        script_gripper = True
        auto_close_dist = 0.40
        open_gripper = 0.0
        closed_gripper = -0.8
        ignore_dog_actions = True
    Cfg.real_grasp = RealGraspCfg


def build_cfg(args):
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)
    add_cfg()

    kp, kd = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = kp
    Cfg.dog.control.damping_leg["joint"] = kd
    Cfg.control.stiffness["joint"] = kp
    Cfg.control.damping["joint"] = kd

    Cfg.asset.fix_base_link = True
    Cfg.env.keep_arm_fixed = False
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
    Cfg.env.episode_length_s = 0.50 if args.debug else 2.40

    Cfg.hybrid.use_vision = True
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
    Cfg.hybrid.reward_scales.orientation_control = 0.0
    Cfg.hybrid.reward_scales.hip_action_l2 = 0.0
    Cfg.hybrid.reward_scales.arm_energy = -5.0e-7
    Cfg.hybrid.reward_scales.arm_dof_vel = -1.0e-5
    Cfg.hybrid.reward_scales.arm_dof_acc = -1.0e-8
    Cfg.hybrid.reward_scales.arm_action_rate = -2.0e-4
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = -5.0e-4
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = -5.0e-4
    Cfg.hybrid.reward_scales.arm_control_smoothness_1 = 0.0
    Cfg.hybrid.reward_scales.arm_control_smoothness_2 = 0.0
    Cfg.hybrid.reward_scales.arm_control_limits = 0.0
    Cfg.hybrid.reward_scales.grasp_obj_dist = 12.0
    Cfg.hybrid.reward_scales.grasp_xy_align = 10.0
    Cfg.hybrid.reward_scales.grasp_z_align = 8.0
    Cfg.hybrid.reward_scales.grasp_close_bonus = 20.0
    Cfg.hybrid.reward_scales.grasp_lift_height = 0.0
    Cfg.hybrid.reward_scales.grasp_drop_penalty = 0.0

    for k in [
        "torques", "dof_vel", "dof_acc", "collision", "action_rate",
        "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel",
        "dof_pos_limits", "feet_slip", "feet_clearance_cmd_linear",
        "action_smoothness_1", "action_smoothness_2", "loco_energy", "jump"
    ]:
        setattr(Cfg.reward_scales, k, 0.0)

    Cfg.rewards.terminal_body_height = 0.01
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.only_positive_rewards = True
    Cfg.asset.render_sphere = True
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"

    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end = 0
    global_switch.init_sigmoid_lr()
    global_switch.open_switch()
    RunnerArgs.save_interval = 50
    PPO_Args.schedule = 'fixed'              # CRITICAL: disable adaptive LR that ramps to 1e-2
    PPO_Args.learning_rate = 1.0e-5           # v7: lower LR to preserve pretrained arm reaching
    PPO_Args.adaptation_module_learning_rate = 1.0e-4  # adaptation can learn faster
    PPO_Args.value_loss_coef = 0.25            # reduced to stabilise critic without tanh bound
    PPO_Args.entropy_coef = 0.03              # v7: higher entropy to prevent premature collapse
    PPO_Args.num_learning_epochs = 4          # more passes for better sample efficiency
    PPO_Args.num_mini_batches = 4             # larger batches for lower variance
    PPO_Args.max_grad_norm = 0.5
    if args.headless:
        RunnerArgs.log_video = False


def train(args):
    mode = "disabled" if (args.no_wandb or args.debug) else ("offline" if args.offline else "online")
    args.seed = set_seed(args.seed)
    build_cfg(args)

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

    log_dir = osp.join(f"{MINI_GYM_ROOT_DIR}/runs/{args.run_name}", wandb.run.name) + f"_seed{args.seed}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(osp.join(log_dir, "checkpoints_arm"), exist_ok=True)
    os.makedirs(osp.join(log_dir, "scripts"), exist_ok=True)
    os.makedirs(osp.join(log_dir, "videos"), exist_ok=True)
    os.makedirs(osp.join(log_dir, "deploy_model"), exist_ok=True)
    os.makedirs(f"{MINI_GYM_ROOT_DIR}/tmp/deploy_model", exist_ok=True)

    if not args.debug:
        shutil.copyfile(__file__, osp.join(log_dir, "scripts", "auto_train_grasp_armonly.py"))
        for extra in ["real_grasp_env.py", "real_grasp_rewards.py"]:
            src = osp.join(MINI_GYM_ROOT_DIR, extra)
            if osp.exists(src):
                shutil.copyfile(src, osp.join(log_dir, "scripts", extra))
        with open(osp.join(log_dir, "params.txt"), "w", encoding="utf-8") as f:
            f.write(str({"Cfg": vars(Cfg), "real_grasp": vars(Cfg.real_grasp)}))
        with open(osp.join(log_dir, "parameters.pkl"), "wb") as f:
            pickle.dump({"note": "arm-only fixed-base config; see params.txt"}, f)

    env = RealGraspEnv(sim_device=args.sim_device, headless=args.headless, cfg=Cfg)
    env = HistoryWrapper(env)
    device = args.sim_device

    if args.debug:
        RunnerArgs.save_interval = 2

    arm_model = ArmActorCritic(
        num_obs=env.cfg.arm.arm_num_observations,
        num_privileged_obs=env.cfg.arm.arm_num_privileged_obs,
        num_obs_history=env.cfg.arm.arm_num_obs_history,
        num_actions=env.cfg.arm.num_actions_arm_cd,
        device=device,
    ).to(device)
    if args.resume:
        arm_model.load_state_dict(torch.load(args.arm_resume_path, map_location=device))
        print("successfully loaded arm weights!!!")

        # CRITICAL: Reset critic (value function) weights.
        # The pretrained value function predicts returns for the original reward
        # structure (command tracking), which is completely wrong for grasp rewards.
        # Keeping it causes bootstrap feedback loops and value loss explosion.
        import torch.nn as nn
        for module in [arm_model.critic_history_encoder, arm_model.critic_body]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)
        print("Reset critic (value function) weights for new reward structure")

    alg_arm = PPO(arm_model, device=device)
    alg_arm.init_storage(
        env.num_train_envs,
        RunnerArgs.num_steps_per_env,
        [env.cfg.arm.arm_num_observations],
        [env.cfg.arm.arm_num_privileged_obs],
        [env.cfg.arm.arm_num_obs_history],
        [env.cfg.arm.num_actions_arm_cd],
        [env.cfg.arm.num_actions_arm_cd],
    )

    env.reset()
    obs_dict_arm = env.get_arm_observations()
    obs_arm = obs_dict_arm["obs"].to(device)
    privileged_obs_arm = obs_dict_arm["privileged_obs"].to(device)
    obs_history_arm = obs_dict_arm["obs_history"].to(device)
    alg_arm.actor_critic.train()

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    step_rewbuffer = deque(maxlen=1000)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    reward_rms = RunningMeanStd(device=device)  # reward normalizer
    ep_infos = []
    tot_timesteps = 0
    tot_time = 0.0

    if args.debug:
        RunnerArgs.save_interval = 2

    for it in range(args.num_learning_iterations):
        start = time.time()
        with torch.inference_mode():
            for i in range(RunnerArgs.num_steps_per_env):
                actions_arm = alg_arm.act(obs_arm[:env.num_train_envs], privileged_obs_arm[:env.num_train_envs], obs_history_arm[:env.num_train_envs])
                env.plan(actions_arm[..., -env.num_plan_actions:])

                zero_dog = torch.zeros(env.num_envs, 12, device=device)
                rewards_dog, rewards_arm, dones, infos = env.step(zero_dog, actions_arm[..., :-env.num_plan_actions])

                obs_dict_arm = env.get_arm_observations()
                obs_arm = obs_dict_arm["obs"].to(device)
                privileged_obs_arm = obs_dict_arm["privileged_obs"].to(device)
                obs_history_arm = obs_dict_arm["obs_history"].to(device)
                rewards_arm = rewards_arm.to(device)
                dones = dones.to(device)

                step_rewbuffer.extend(rewards_arm[:env.num_train_envs].detach().cpu().numpy().tolist())

                alg_arm.process_env_step(rewards_arm[:env.num_train_envs], dones[:env.num_train_envs], infos)
                env_ids = dones.nonzero(as_tuple=False).flatten()
                env.clear_cached(env_ids)

                if "train/episode" in infos:
                    ep_infos.append(infos["train/episode"])

                cur_reward_sum += rewards_arm
                cur_episode_length += 1
                new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                new_ids_train = new_ids[new_ids < env.num_train_envs]
                rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                cur_reward_sum[new_ids_train] = 0
                cur_episode_length[new_ids_train] = 0

            collection_time = time.time() - start
            start_learn = time.time()
            alg_arm.compute_returns(obs_history_arm[:env.num_train_envs], privileged_obs_arm[:env.num_train_envs])

        mean_value_loss_arm, mean_surrogate_loss_arm, mean_adaptation_module_loss_arm, *_ = alg_arm.update(un_adapt=False)
        learn_time = time.time() - start_learn
        tot_time += collection_time + learn_time
        tot_timesteps += RunnerArgs.num_steps_per_env * env.num_envs
        fps = RunnerArgs.num_steps_per_env * env.num_envs / max(collection_time + learn_time, 1e-6)

        ep_string = ""
        if ep_infos:
            for key in ep_infos[0].keys():
                mean = torch.mean(torch.stack([ep_info[key] for ep_info in ep_infos]))
                ep_string += f"{('Mean episode ' + key + ':'):>35} {mean:.4f}\n"

        log_string = (
            f"{'#' * 80}\n"
            f"{(' Learning iteration ' + str(it) + '/' + str(args.num_learning_iterations) + ' ').center(80, ' ')}\n\n"
            f"{ep_string}"
            f"{'-' * 80}\n"
            f"{'run_name:':>35} {args.run_name}\n"
            f"{'Computation:':>35} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
            f"{'Arm Value function loss:':>35} {mean_value_loss_arm:.8f}\n"
            f"{'Arm Surrogate loss:':>35} {mean_surrogate_loss_arm:.8f}\n"
            f"{'Arm Adaptation loss:':>35} {mean_adaptation_module_loss_arm:.8f}\n"
            f"{'Mean step reward (arm):':>35} {statistics.mean(step_rewbuffer) if step_rewbuffer else 0.0:.4f}\n"
            f"{'Mean reward (arm):':>35} {statistics.mean(rewbuffer) if rewbuffer else 0.0:.4f}\n"
            f"{'Mean episode length:':>35} {statistics.mean(lenbuffer) if lenbuffer else 0.0:.4f}\n"
            f"{'Total timesteps:':>35} {tot_timesteps}\n"
            f"{'Total time:':>35} {tot_time:.2f}s\n"
        )
        print(log_string)
        with open(osp.join(log_dir, "log.txt"), "a") as f:
            f.write(log_string)

        # Diagnostic: log actual distances every 50 iterations
        if it % 50 == 0:
            with torch.inference_mode():
                gp = env.grasper_pos_world()
                op = env.object_spawn_pos
                d = torch.norm(gp - op, dim=1)
                bp = env.root_states[:env.num_envs, :3]
                diag = (
                    f"  [DIAG iter {it}] base_z={bp[:,2].mean():.3f} "
                    f"grasper=({gp[:,0].mean():.3f},{gp[:,1].mean():.3f},{gp[:,2].mean():.3f}) "
                    f"object=({op[:,0].mean():.3f},{op[:,1].mean():.3f},{op[:,2].mean():.3f}) "
                    f"dist={d.mean():.4f} min={d.min():.4f} max={d.max():.4f}\n"
                )
                print(diag)
                with open(osp.join(log_dir, "log.txt"), "a") as f:
                    f.write(diag)

        if not args.debug and it % RunnerArgs.save_interval == 0:
            torch.save(arm_model.state_dict(), osp.join(log_dir, "checkpoints_arm", f"ac_weights_{it:06d}.pt"))
            torch.save(arm_model.state_dict(), osp.join(log_dir, "checkpoints_arm", "ac_weights_last_arm.pt"))
        ep_infos.clear()

    torch.save(arm_model.state_dict(), osp.join(log_dir, "checkpoints_arm", "ac_weights_last_arm.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm-only fixed-base B2Z1 reach pretraining")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--num_learning_iterations", type=int, default=5000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--run_name", type=str, default="b2z1_grasp_armonly_v8")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm_resume_path", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--tags", nargs="+", default=[])
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    train(parser.parse_args())
