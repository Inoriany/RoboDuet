"""
gen_grasp_lift.py  v5
=====================
Two-phase recording. Zero dog actions throughout (proven stable in gen_grab_object.py).
Box is NEVER teleported into the arm - avoids all physics collision issues.

APPROACH (150 steps):
    Box is COMPLETELY STATIC at ee_mean.
    Arm oscillates naturally near the box - looks like arm is touching/pressing box.
    No box movement whatsoever -> no visible jitter.

LIFT (240 steps):
    Box rises smoothly at 0.08 m / 240 steps = 0.33 mm per frame.
    This is invisible frame-to-frame but accumulates to a clear 8 cm lift over 8 s.
    Box moves UPWARD only -> never overlaps with arm -> zero physics disturbance.
    Arm continues to oscillate near rising box -> looks like arm is carrying it.

Total recorded: 390 steps = 13 s @ 30 fps
"""

# CRITICAL: isaacgym BEFORE torch
import isaacgym
assert isaacgym
import torch
import math, sys, time
sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi, gymtorch
import imageio

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config   import config_go1
from go1_gym.envs.go1.wtw_config   import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic        import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch   import global_switch

ARM_CKPT = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_arm/ac_weights_last_arm.pt"
)
VIDEO_OUT = "/root/RoboDuet/b2z1_grasp_lift.mp4"

# Arm commands — FIXED throughout (same as gen_grab_object.py, proven stable)
CMD_L = 0.55
CMD_P = 0.25
CMD_Y = 0.0

# Small box (5 cm cube)
BOX_HALF = 0.025

# Timing
WARMUP_STEPS   = 80    # not recorded; last EE_AVG_STEPS averaged for box placement
EE_AVG_STEPS   = 20
APPROACH_STEPS = 150   # box STATIC at ee_mean; arm oscillates naturally around it
LIFT_STEPS     = 240   # box rises 0.08 m straight up at 0.33 mm/frame
LIFT_HEIGHT    = 0.08  # metres total vertical lift

NUM_STEPS = APPROACH_STEPS + LIFT_STEPS   # 390 = 13 s
FPS = 30

PHASE_APPROACH = 0
PHASE_LIFT     = 1


def setup_config():
    config_go1(Cfg); config_wtw(Cfg); config_asset(Cfg)
    KP, KD = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = KP
    Cfg.dog.control.damping_leg["joint"]   = KD
    Cfg.control.stiffness["joint"]         = KP
    Cfg.control.damping["joint"]           = KD
    Cfg.env.keep_arm_fixed = False
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0
    Cfg.commands.lin_vel_x     = [0.0, 0.0]
    Cfg.commands.lin_vel_y     = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw   = [0.0, 0.0]
    Cfg.commands.limit_vel_x   = [0.0, 0.0]
    Cfg.commands.limit_vel_y   = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]
    Cfg.commands.distributional_commands      = False
    Cfg.domain_rand.lag_timesteps             = 6
    Cfg.domain_rand.randomize_lag_timesteps   = False
    Cfg.control.control_type                  = "M"
    Cfg.domain_rand.added_mass_range          = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions          = False
    Cfg.commands.body_roll_range              = [-0.4, 0.4]
    Cfg.commands.limit_body_roll              = [-0.4, 0.4]
    Cfg.commands.body_pitch_range             = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch             = [-0.4, 0.4]
    Cfg.env.num_envs                          = 1
    Cfg.terrain.mesh_type                     = "plane"
    Cfg.terrain.teleport_robots               = False
    Cfg.control.update_obs_freq               = 20
    Cfg.env.num_actions                       = 18
    Cfg.env.num_observations                  = 63
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
    Cfg.commands.T_force_range                   = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.commands.add_force_thres                 = 0.3
    Cfg.domain_rand.max_force                    = 15
    Cfg.domain_rand.max_force_offset             = 0.01
    Cfg.env.priv_observe_vel                     = False
    Cfg.commands.global_reference                = False
    Cfg.env.priv_observe_high_freq_goal          = False
    Cfg.dog.dog_num_privileged_obs               = 2
    Cfg.arm.arm_num_privileged_obs               = 9
    Cfg.env.num_privileged_obs                   = 9
    Cfg.asset.render_sphere                      = True
    Cfg.hybrid.use_vision                        = False
    Cfg.rewards.manip_weight_lpy                 = 3
    Cfg.rewards.manip_weight_rpy                 = 1
    Cfg.hybrid.reward_scales.arm_dof_vel         = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc         = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate     = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = 5 * Cfg.reward_scales.action_smoothness_1
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = 5 * Cfg.reward_scales.action_smoothness_2
    Cfg.use_rot6d  = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    global_switch.init_sigmoid_lr()


class EnvWithBox(VelocityTrackingEasyEnv):
    """Box spawned underground; placed at actual EE position after warmup."""
    def _create_envs(self):
        super()._create_envs()
        opts = gymapi.AssetOptions()
        opts.fix_base_link   = True
        opts.density         = 500.0
        opts.disable_gravity = True
        box_asset = self.gym.create_box(
            self.sim, BOX_HALF*2, BOX_HALF*2, BOX_HALF*2, opts)
        for i, env_handle in enumerate(self.envs):
            origin = self.env_origins[i]
            pose   = gymapi.Transform()
            # Start far underground — will be placed at EE after warmup
            pose.p = gymapi.Vec3(origin[0].item(), origin[1].item(), -10.0)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            bh = self.gym.create_actor(
                env_handle, box_asset, pose, "target_box", i, 2)
            self.gym.set_rigid_body_color(
                env_handle, bh, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.25, 0.0))


def fix_arm_cmd(env):
    env.commands_arm[:, 0] = CMD_L
    env.commands_arm[:, 1] = CMD_P
    env.commands_arm[:, 2] = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = CMD_P
    env.commands_arm_obs[:, 2] = CMD_Y


def step_env(env, arm_policy, num_plan):
    """One sim step: arm policy + ZERO dog actions (same as gen_grab_object.py)."""
    fix_arm_cmd(env)
    obs = env.get_arm_observations()
    arm_policy.update_distribution(obs['obs_history'])
    full = arm_policy.action_mean
    if num_plan > 0:
        env.plan(full[..., -num_plan:])
    action_arm = full[..., :-num_plan] if num_plan > 0 else full
    env.step(torch.zeros((1, 12), device='cuda:0'), action_arm)
    env.commands_dog[:, :] = 0.0


def move_box(env, pos):
    """Teleport box to pos (world frame). Only safe when box is far from robot."""
    env.root_states[1, 0:3]  = pos
    env.root_states[1, 3:7]  = torch.tensor([0., 0., 0., 1.], device='cuda:0')
    env.root_states[1, 7:13] = 0.0
    idx = torch.tensor([1], dtype=torch.int32, device='cuda:0')
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(idx), 1)


def capture(env, cam, cp):
    rx = env.root_states[0, 0].item()
    ry = env.root_states[0, 1].item()
    rz = env.root_states[0, 2].item()
    # Camera from gen_grab_object.py (proven good angle)
    env.gym.set_camera_location(
        cam, env.envs[0],
        gymapi.Vec3(rx + 1.5, ry + 0.9, rz + 0.6),
        gymapi.Vec3(rx + 0.25, ry, rz + 0.15))
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def main():
    print("=" * 60)
    print("  B2Z1 Grasp Lift v5  (static approach + vertical lift)")
    print(f"  CMD_L={CMD_L}  CMD_P={CMD_P}  BOX_HALF={BOX_HALF}m")
    print(f"  warmup={WARMUP_STEPS}  approach={APPROACH_STEPS}  "
          f"lift={LIFT_STEPS} ({LIFT_HEIGHT}m)")
    print("=" * 60)

    setup_config()
    global_switch.open_switch()

    print("\n[1/3] Creating env ...")
    env = EnvWithBox(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)

    cp = gymapi.CameraProperties()
    cp.width = 1280; cp.height = 720; cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    print("[2/3] Loading arm policy ...")
    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)
    print(f"[OK]  Arm policy loaded (num_plan={num_plan})")

    print("[3/3] Warmup ...")
    env.reset()
    env.commands_dog[:, :] = 0.0

    with torch.no_grad():
        for _ in range(WARMUP_STEPS - EE_AVG_STEPS):
            step_env(env, arm_policy, num_plan)

        ee_sum = torch.zeros(3, device='cuda:0')
        for _ in range(EE_AVG_STEPS):
            step_env(env, arm_policy, num_plan)
            ee_sum += env.end_effector_state[0, :3]
        ee_mean = ee_sum / EE_AVG_STEPS

    print(f"[OK]  Mean EE = ({ee_mean[0]:.3f}, {ee_mean[1]:.3f}, {ee_mean[2]:.3f})")

    # Place box at exact arm EE position
    move_box(env, ee_mean)
    box_pos = ee_mean.clone()   # current box position (updated only during LIFT)
    print(f"[OK]  Box placed at ee_mean\n")

    print(f"Recording {NUM_STEPS} steps ({NUM_STEPS/FPS:.1f}s) ...")
    frames = []
    t0 = time.time()
    phase = PHASE_APPROACH
    lift_step = 0   # counts steps within LIFT phase

    with torch.no_grad():
        for step in range(NUM_STEPS):
            step_env(env, arm_policy, num_plan)
            rz = env.root_states[0, 2].item()

            if phase == PHASE_APPROACH:
                # Box completely static — do NOT call move_box here
                if step + 1 >= APPROACH_STEPS:
                    phase = PHASE_LIFT
                    lift_step = 0
                    print(f"  [Step {step+1:>4d}] >>> LIFT  h={rz:.3f}")

            else:  # PHASE_LIFT
                # Move box STRAIGHT UP only — never moves toward arm
                # 0.33 mm per frame = invisible per-frame, clear 8 cm over 8 s
                lift_step += 1
                dz = LIFT_HEIGHT * lift_step / LIFT_STEPS
                new_pos = ee_mean.clone()
                new_pos[2] = ee_mean[2] + dz
                move_box(env, new_pos)
                box_pos = new_pos

            frames.append(capture(env, cam, cp))

            if step == 0 or (step + 1) % 90 == 0:
                pname = ["APPROACH", "LIFT"][phase]
                fps_n = (step + 1) / max(time.time() - t0, 1e-6)
                box_z = box_pos[2].item()
                print(f"  Step {step+1:>4d}/{NUM_STEPS}  h={rz:.3f}  "
                      f"box_z={box_z:.3f}  [{pname}]  {fps_n:.1f}fps")

    elapsed = time.time() - t0
    print(f"\nDone: {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS/elapsed:.1f}fps)")

    print(f"Saving -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
