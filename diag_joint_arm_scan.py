import isaacgym
assert isaacgym
import torch
import sys

sys.path.insert(0, '/root/RoboDuet')

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym.utils.global_switch import global_switch


def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)
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
    for k in [
        'randomize_friction', 'randomize_base_mass', 'randomize_restitution',
        'randomize_com_displacement', 'randomize_motor_strength',
        'randomize_motor_offset', 'randomize_gravity', 'push_robots',
        'randomize_end_effector_force'
    ]:
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


def step_env(env, arm_action):
    dog_action = torch.zeros((1, 12), device='cuda:0')
    env.step(dog_action, arm_action.unsqueeze(0))
    env.commands_dog[:, :] = 0.0


def main():
    setup_config()
    global_switch.open_switch()
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)

    candidates = [
        [0.0, -4.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, -3.5, 3.5, 0.0, 0.0, 0.0],
        [0.0, -3.0, 3.0, 0.0, 0.0, 0.0],
        [0.5, -3.5, 3.5, 0.0, 0.0, 0.0],
        [-0.5, -3.5, 3.5, 0.0, 0.0, 0.0],
        [0.0, -4.5, 4.2, -0.5, 0.0, 0.0],
        [0.0, -4.2, 4.0, -0.8, 0.0, 0.0],
        [0.0, -3.8, 3.6, -0.8, 0.0, 0.0],
        [0.0, -3.5, 3.2, -1.0, 0.0, 0.0],
        [0.0, -2.8, 2.6, -1.0, 0.0, 0.0],
    ]

    print("idx | base_h | ee(x,y,z) | gm(x,y,z) | action")
    print("-" * 120)
    with torch.no_grad():
        for idx, cand in enumerate(candidates):
            env.reset()
            env.commands_dog[:, :] = 0.0
            start = torch.zeros(6, device='cuda:0')
            target = torch.tensor(cand, device='cuda:0')
            for i in range(160):
                frac = min(1.0, (i + 1) / 90.0)
                arm = start + (target - start) * frac
                step_env(env, arm)
            for _ in range(30):
                step_env(env, target)
            ee = env.end_effector_state[0, :3].cpu().tolist()
            bodies = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
            gm = bodies[0, 25, :3].cpu().tolist()
            h = env.root_states[0, 2].item()
            print(
                f"{idx:>3d} | {h:>.3f} | "
                f"({ee[0]:>.3f},{ee[1]:>.3f},{ee[2]:>.3f}) | "
                f"({gm[0]:>.3f},{gm[1]:>.3f},{gm[2]:>.3f}) | {cand}"
            )


if __name__ == "__main__":
    main()
