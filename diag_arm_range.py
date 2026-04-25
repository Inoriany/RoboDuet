"""
diag_arm_range.py — Diagnostic: measure EE position & stability at different CMD_L/CMD_P
Outputs a table of (cmd_l, cmd_p) -> (base_h, ee_avg_x, ee_avg_y, ee_avg_z, stable?)
"""
import isaacgym
assert isaacgym
import torch
import sys, time

sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymtorch
from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch import global_switch

ARM_CKPT = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_arm/ac_weights_last_arm.pt"
)

def setup_config():
    config_go1(Cfg); config_wtw(Cfg); config_asset(Cfg)
    Cfg.dog.control.stiffness_leg["joint"] = 200.0
    Cfg.dog.control.damping_leg["joint"] = 20.0
    Cfg.control.stiffness["joint"] = 200.0
    Cfg.control.damping["joint"] = 20.0
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
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
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
    for k in ['randomize_friction','randomize_base_mass','randomize_restitution',
              'randomize_com_displacement','randomize_motor_strength',
              'randomize_motor_offset','randomize_gravity','push_robots',
              'randomize_end_effector_force']:
        setattr(Cfg.domain_rand, k, False)
    Cfg.commands.T_force_range = [2, 4.0]
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


def main():
    setup_config()
    global_switch.open_switch()
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)

    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)

    def step(cmd_l, cmd_p):
        env.commands_arm[:, 0] = cmd_l
        env.commands_arm[:, 1] = cmd_p
        env.commands_arm[:, 2] = 0.0
        env.commands_arm_obs[:, 0] = cmd_l
        env.commands_arm_obs[:, 1] = cmd_p
        env.commands_arm_obs[:, 2] = 0.0
        arm_obs = env.get_arm_observations()
        arm_policy.update_distribution(arm_obs['obs_history'])
        arm_full = arm_policy.action_mean
        if num_plan > 0:
            env.plan(arm_full[..., -num_plan:])
        a_arm = arm_full[..., :-num_plan] if num_plan > 0 else arm_full
        a_dog = torch.zeros((1, 12), device='cuda:0')
        env.step(a_dog, a_arm)
        env.commands_dog[:, :] = 0.0

    # Test configurations:
    # First: varying CMD_L at fixed CMD_P=0.25
    # Then: varying CMD_P at fixed CMD_L=0.55
    tests = []
    for l in [0.40, 0.45, 0.48, 0.50, 0.53, 0.55, 0.58, 0.60, 0.65]:
        tests.append((l, 0.25))
    for p in [0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]:
        tests.append((0.55, p))

    print(f"\n{'CMD_L':>6} {'CMD_P':>6} | {'base_h':>7} {'ee_x':>7} {'ee_y':>7} {'ee_z':>7} | {'ee_x_rng':>10} {'ee_z_rng':>10} | {'stable':>6}")
    print("-" * 95)

    with torch.no_grad():
        for cmd_l, cmd_p in tests:
            # Reset env for each test to avoid cumulative drift
            env.reset()
            env.commands_dog[:, :] = 0.0

            # Warmup: start at known-good, gradually ramp
            for i in range(100):
                frac = min(1.0, i / 60.0)
                l_now = 0.55 + (cmd_l - 0.55) * frac
                p_now = 0.25 + (cmd_p - 0.25) * frac
                step(l_now, p_now)

            # Settle at target
            for _ in range(50):
                step(cmd_l, cmd_p)

            # Sample EE
            ee_samples = []
            h_samples = []
            for _ in range(25):
                step(cmd_l, cmd_p)
                ee = env.end_effector_state[0, :3].cpu().tolist()
                h = env.root_states[0, 2].item()
                ee_samples.append(ee)
                h_samples.append(h)

            n = len(ee_samples)
            avg = [sum(s[i] for s in ee_samples)/n for i in range(3)]
            mn = [min(s[i] for s in ee_samples) for i in range(3)]
            mx = [max(s[i] for s in ee_samples) for i in range(3)]
            avg_h = sum(h_samples) / n
            stable = "YES" if avg_h > 0.35 else ("WARN" if avg_h > 0.25 else "FALL")

            print(f"{cmd_l:>6.2f} {cmd_p:>6.2f} | {avg_h:>7.3f} {avg[0]:>7.3f} {avg[1]:>7.3f} {avg[2]:>7.3f} | "
                  f"[{mn[0]:.2f},{mx[0]:.2f}] [{mn[2]:.2f},{mx[2]:.2f}] | {stable:>6}")

    print("\nDone.")


if __name__ == "__main__":
    main()
