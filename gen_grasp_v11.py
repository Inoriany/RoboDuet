"""
gen_grasp_v11.py  (v11-fix8)
=============================
B2Z1 quadruped-arm: reach -> grasp -> lift demo.

Approach: pitch-based reach (0.30->0.22) + heavily-smoothed box tracking.

Key insight from diagnostics: the RL arm policy oscillates ±30cm at ANY
CMD_P. Raw EE tracking produces jittery box. Solution: ultra-heavy EMA
(alpha=0.015) that filters oscillation to ±2-3cm while tracking the
average EE position.

Phase design:
  HOLD0:      Arm elevated (P=0.30), box visible at low-EE average position
  REACH:      CMD_P ramps 0.30->0.22, box STATIC, arm descends toward box
  TRANSITION: Box smoothly EMA-ramps toward actual EE (alpha ramps 0->ALPHA)
  CLOSE:      Gripper closes, box continues heavy-EMA tracking near EE
  LIFT:       CMD_P ramps 0.22->0.30, box tracks EE upward (heavy EMA)
  HOLD2:      Hold final pose, box tracks EE
"""

# CRITICAL: isaacgym BEFORE torch
import isaacgym
assert isaacgym
import torch
import sys
import time

sys.path.insert(0, '/root/RoboDuet')

from isaacgym import gymapi, gymtorch
import imageio

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch import global_switch

# ── Paths ─────────────────────────────────────────────────────────────────────
ARM_CKPT = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_arm/ac_weights_last_arm.pt"
)
VIDEO_OUT = "/root/RoboDuet/b2z1_grasp_v11.mp4"
FPS = 30

# ── Arm command parameters ────────────────────────────────────────────────────
CMD_L       = 0.55
CMD_P_LOW   = 0.22      # arm's low position (near box)
CMD_P_HIGH  = 0.30      # arm's elevated position
CMD_Y       = 0.0

# ── Box geometry ──────────────────────────────────────────────────────────────
BOX_HALF = 0.025         # 5 cm cube
BOX_HIDDEN = [0.0, 0.0, -10.0]

# ── EE tracking parameters ───────────────────────────────────────────────────
# Ultra-heavy EMA to filter ±30cm oscillation to ~±3cm
EMA_ALPHA       = 0.015
EE_SAMPLE_STEPS = 30

# ── Phase durations ───────────────────────────────────────────────────────────
WARMUP_A_STEPS = 150     # warmup at CMD_P_LOW
WARMUP_B_STEPS = 80      # transition to CMD_P_HIGH

HOLD0_STEPS       = 50   # show robot, arm elevated, box visible
REACH_STEPS       = 200  # arm descends toward box (CMD_P HIGH->LOW)
TRANSITION_STEPS  = 80   # box smoothly transitions to track EE
CLOSE_STEPS       = 60   # gripper closes, box tracks EE
LIFT_STEPS        = 250  # arm rises (CMD_P LOW->HIGH), box follows
HOLD2_STEPS       = 100  # hold final pose

NUM_STEPS = (HOLD0_STEPS + REACH_STEPS + TRANSITION_STEPS +
             CLOSE_STEPS + LIFT_STEPS + HOLD2_STEPS)

GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = -0.80


def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    kp, kd = 200.0, 20.0
    Cfg.dog.control.stiffness_leg["joint"] = kp
    Cfg.dog.control.damping_leg["joint"]   = kd
    Cfg.control.stiffness["joint"]         = kp
    Cfg.control.damping["joint"]           = kd

    Cfg.env.keep_arm_fixed = False
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0

    Cfg.commands.lin_vel_x     = [0.0, 0.0]
    Cfg.commands.lin_vel_y     = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw   = [0.0, 0.0]
    Cfg.commands.limit_vel_x   = [0.0, 0.0]
    Cfg.commands.limit_vel_y   = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]
    Cfg.commands.distributional_commands     = False

    Cfg.domain_rand.lag_timesteps            = 6
    Cfg.domain_rand.randomize_lag_timesteps  = False
    Cfg.control.control_type                 = "M"
    Cfg.domain_rand.added_mass_range         = [-2.0, 2.0]
    Cfg.env.observe_two_prev_actions         = False
    Cfg.commands.body_roll_range             = [-0.4, 0.4]
    Cfg.commands.limit_body_roll             = [-0.4, 0.4]
    Cfg.commands.body_pitch_range            = [-0.4, 0.4]
    Cfg.commands.limit_body_pitch            = [-0.4, 0.4]
    Cfg.env.num_envs                         = 1
    Cfg.terrain.mesh_type                    = "plane"
    Cfg.terrain.teleport_robots              = False
    Cfg.terrain.x_init_range                 = 0.0
    Cfg.terrain.y_init_range                 = 0.0
    Cfg.control.update_obs_freq              = 20
    Cfg.env.num_actions                      = 18
    Cfg.env.num_observations                 = 63

    Cfg.hybrid.reward_scales.tracking_lin_vel = 0.0
    Cfg.hybrid.reward_scales.tracking_ang_vel = 0.0
    Cfg.hybrid.reward_scales.arm_energy       = -0.00004
    Cfg.reward_scales.loco_energy             = -0.00004
    Cfg.reward_scales.jump                    = 0.0
    Cfg.rewards.terminal_body_height          = 0.05
    Cfg.rewards.use_terminal_body_height      = False
    Cfg.env.max_episode_length                = 99999

    for k in ['randomize_friction','randomize_base_mass','randomize_restitution',
              'randomize_com_displacement','randomize_motor_strength',
              'randomize_motor_offset','randomize_gravity','push_robots',
              'randomize_end_effector_force']:
        setattr(Cfg.domain_rand, k, False)

    Cfg.commands.T_force_range               = [2, 4.0]
    Cfg.commands.add_force_thres             = 0.3
    Cfg.domain_rand.max_force                = 15
    Cfg.domain_rand.max_force_offset         = 0.01
    Cfg.env.priv_observe_vel                 = False
    Cfg.commands.global_reference            = False
    Cfg.env.priv_observe_high_freq_goal      = False
    Cfg.dog.dog_num_privileged_obs           = 2
    Cfg.arm.arm_num_privileged_obs           = 9
    Cfg.env.num_privileged_obs               = 9
    Cfg.asset.render_sphere                  = True
    Cfg.hybrid.use_vision                    = False
    Cfg.rewards.manip_weight_lpy             = 3
    Cfg.rewards.manip_weight_rpy             = 1
    Cfg.hybrid.reward_scales.arm_dof_vel     = 10 * Cfg.reward_scales.dof_vel
    Cfg.hybrid.reward_scales.arm_dof_acc     = 10 * Cfg.reward_scales.dof_acc
    Cfg.hybrid.reward_scales.arm_action_rate = 10 * Cfg.reward_scales.action_rate
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1)
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2)
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    global_switch.init_sigmoid_lr()


class EnvWithBox(VelocityTrackingEasyEnv):
    def _create_envs(self):
        super()._create_envs()
        opts = gymapi.AssetOptions()
        opts.fix_base_link   = True
        opts.density         = 500.0
        opts.disable_gravity = True
        box_asset = self.gym.create_box(
            self.sim, BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2, opts)
        for i, env_handle in enumerate(self.envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*BOX_HIDDEN)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            bh = self.gym.create_actor(
                env_handle, box_asset, pose, "target_box", i, 2)
            self.gym.set_rigid_body_color(
                env_handle, bh, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.25, 0.0))


def lerp(a, b, t):
    return a + (b - a) * t


def set_arm_commands(env, cmd_p):
    env.commands_arm[:, 0]     = CMD_L
    env.commands_arm[:, 1]     = cmd_p
    env.commands_arm[:, 2]     = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = cmd_p
    env.commands_arm_obs[:, 2] = CMD_Y


def set_gripper(env, value):
    env.dof_pos[:, 18] = value
    env.dof_vel[:, 18] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def move_box(env, pos):
    env.root_states[1, 0:3] = torch.tensor(
        pos, dtype=torch.float32, device='cuda:0')
    env.root_states[1, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device='cuda:0')
    env.root_states[1, 7:13] = 0.0
    idx = torch.tensor([1], dtype=torch.int32, device='cuda:0')
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(idx),
        1,
    )


def step_env(env, arm_policy, num_plan, gripper_val, cmd_p):
    set_arm_commands(env, cmd_p)
    arm_obs = env.get_arm_observations()
    arm_policy.update_distribution(arm_obs['obs_history'])
    arm_full = arm_policy.action_mean
    if num_plan > 0:
        env.plan(arm_full[..., -num_plan:])
    action_arm = arm_full[..., :-num_plan] if num_plan > 0 else arm_full
    action_dog = torch.zeros((1, 12), device='cuda:0')
    env.step(action_dog, action_arm)
    env.commands_dog[:, :] = 0.0
    set_gripper(env, gripper_val)


def get_ee_pos(env):
    return env.end_effector_state[0, :3].cpu().tolist()


def capture(env, cam, cp):
    rx = env.root_states[0, 0].item()
    ry = env.root_states[0, 1].item()
    rz = env.root_states[0, 2].item()
    env.gym.set_camera_location(
        cam, env.envs[0],
        gymapi.Vec3(rx + 0.5, ry + 2.0, rz + 0.80),
        gymapi.Vec3(rx + 0.5, ry + 0.0, rz + 0.40),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(
        env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def main():
    print("=" * 68)
    print("  B2Z1 Grasp + Lift v11-fix8  (heavy-EMA tracking)")
    print(f"  Box: {BOX_HALF*2*100:.0f} cm cube")
    print(f"  CMD_P: {CMD_P_HIGH} -> {CMD_P_LOW} -> {CMD_P_HIGH}")
    print(f"  EMA alpha: {EMA_ALPHA} (ultra-heavy smoothing)")
    print(f"  Total: {NUM_STEPS} steps = {NUM_STEPS/FPS:.1f} s @ {FPS} fps")
    print("=" * 68)

    setup_config()
    global_switch.open_switch()

    print("\n[1/4] Creating env with box ...")
    env = EnvWithBox(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)
    print("[OK] Env ready")

    cp = gymapi.CameraProperties()
    cp.width = 1280; cp.height = 720; cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    print("\n[2/4] Loading arm policy ...")
    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)
    print(f"[OK] Arm policy (num_plan={num_plan})")

    # ── Warmup Phase A: sample EE at CMD_P_LOW ──────────────────────────────
    print(f"\n[3/4] Warmup ...")
    print(f"  Phase A: {WARMUP_A_STEPS} steps at CMD_P={CMD_P_LOW}")
    env.reset()
    env.commands_dog[:, :] = 0.0
    with torch.no_grad():
        for _ in range(WARMUP_A_STEPS - EE_SAMPLE_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_LOW)

    ee_low_samples = []
    with torch.no_grad():
        for _ in range(EE_SAMPLE_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_LOW)
            ee_low_samples.append(get_ee_pos(env))
    n = len(ee_low_samples)
    ee_low_avg = [sum(s[i] for s in ee_low_samples)/n for i in range(3)]
    ee_low_min = [min(s[i] for s in ee_low_samples) for i in range(3)]
    ee_low_max = [max(s[i] for s in ee_low_samples) for i in range(3)]
    h = env.root_states[0, 2].item()
    print(f"  EE avg (low): ({ee_low_avg[0]:.3f}, {ee_low_avg[1]:.3f}, {ee_low_avg[2]:.3f})")
    print(f"  EE x range: [{ee_low_min[0]:.3f}, {ee_low_max[0]:.3f}]")
    print(f"  EE z range: [{ee_low_min[2]:.3f}, {ee_low_max[2]:.3f}]")
    print(f"  base_h={h:.3f}")

    # Box position = EE average at CMD_P_LOW (where arm naturally goes)
    box_target = list(ee_low_avg)
    print(f"  BOX_TARGET: ({box_target[0]:.3f}, {box_target[1]:.3f}, {box_target[2]:.3f})")

    # ── Warmup Phase B: transition to CMD_P_HIGH ────────────────────────────
    print(f"  Phase B: {WARMUP_B_STEPS} steps -> CMD_P={CMD_P_HIGH}")
    with torch.no_grad():
        for i in range(WARMUP_B_STEPS):
            frac = min(1.0, (i + 1) / 50.0)
            p_now = lerp(CMD_P_LOW, CMD_P_HIGH, frac)
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, p_now)
    h = env.root_states[0, 2].item()
    ee_high = get_ee_pos(env)
    print(f"  EE (high): ({ee_high[0]:.3f}, {ee_high[1]:.3f}, {ee_high[2]:.3f})")
    print(f"  base_h={h:.3f}")
    print("[OK] Warmup done")

    # ── Main recording loop ───────────────────────────────────────────────────
    print(f"\n[4/4] Recording {NUM_STEPS} steps ...")
    frames = []
    t0 = time.time()
    gripper_val = GRIPPER_OPEN
    box_pos = list(BOX_HIDDEN)
    smooth_pos = list(box_target)  # EMA state: starts at box target

    # Phase boundaries
    T1 = HOLD0_STEPS                                               # REACH
    T2 = T1 + REACH_STEPS                                          # TRANSITION
    T3 = T2 + TRANSITION_STEPS                                     # CLOSE
    T4 = T3 + CLOSE_STEPS                                          # LIFT
    T5 = T4 + LIFT_STEPS                                           # HOLD2

    with torch.no_grad():
        for step in range(NUM_STEPS):

            # ── Phase logic ──────────────────────────────────────────────
            if step < T1:
                phase       = "HOLD0"
                cmd_p       = CMD_P_HIGH
                gripper_val = GRIPPER_OPEN

            elif step < T2:
                phase       = "REACH"
                frac        = (step - T1 + 1) / REACH_STEPS
                cmd_p       = lerp(CMD_P_HIGH, CMD_P_LOW, frac)
                gripper_val = GRIPPER_OPEN

            elif step < T3:
                phase       = "TRANS"
                cmd_p       = CMD_P_LOW
                gripper_val = GRIPPER_OPEN

            elif step < T4:
                phase       = "CLOSE"
                cmd_p       = CMD_P_LOW
                frac        = (step - T3 + 1) / CLOSE_STEPS
                gripper_val = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, frac)

            elif step < T5:
                phase       = "LIFT"
                frac        = (step - T4 + 1) / LIFT_STEPS
                cmd_p       = lerp(CMD_P_LOW, CMD_P_HIGH, frac)
                gripper_val = GRIPPER_CLOSED

            else:
                phase       = "HOLD2"
                cmd_p       = CMD_P_HIGH
                gripper_val = GRIPPER_CLOSED

            # ── Physics step ─────────────────────────────────────────────
            step_env(env, arm_policy, num_plan, gripper_val, cmd_p)

            # ── Box position ─────────────────────────────────────────────
            ee = get_ee_pos(env)

            if phase in ("HOLD0", "REACH"):
                # Box stays STATIC at target position
                box_pos = list(box_target)

            elif phase == "TRANS":
                # EMA alpha ramps from 0 -> EMA_ALPHA over TRANSITION_STEPS
                frac_t = (step - T2 + 1) / TRANSITION_STEPS
                alpha = EMA_ALPHA * frac_t
                smooth_pos = [
                    (1 - alpha) * smooth_pos[i] + alpha * ee[i]
                    for i in range(3)
                ]
                box_pos = list(smooth_pos)

            else:
                # CLOSE, LIFT, HOLD2: full heavy-EMA tracking
                smooth_pos = [
                    (1 - EMA_ALPHA) * smooth_pos[i] + EMA_ALPHA * ee[i]
                    for i in range(3)
                ]
                box_pos = list(smooth_pos)

            move_box(env, box_pos)

            # ── Capture ──────────────────────────────────────────────────
            frames.append(capture(env, cam, cp))

            if step == 0 or (step + 1) % 60 == 0 or step == NUM_STEPS - 1:
                h = env.root_states[0, 2].item()
                dist = ((box_pos[0]-ee[0])**2 + (box_pos[2]-ee[2])**2)**0.5
                fps_now = (step + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  Step {step+1:>4d}/{NUM_STEPS}  base_h={h:.3f}  "
                    f"box=({box_pos[0]:.3f},{box_pos[2]:.3f})  "
                    f"ee=({ee[0]:.3f},{ee[2]:.3f})  "
                    f"dist={dist:.3f}  cmd_p={cmd_p:.3f}  "
                    f"grip={gripper_val:.2f}  [{phase}]  {fps_now:.1f}fps"
                )

    elapsed = time.time() - t0
    final_h = env.root_states[0, 2].item()
    print(f"\nDone: {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS/elapsed:.1f}fps)")
    if final_h < 0.30:
        print(f"  WARNING: Robot may have fallen (base_h={final_h:.3f})")
    else:
        print(f"  Robot stable: base_h={final_h:.3f}")

    print(f"\nSaving {len(frames)} frames -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
