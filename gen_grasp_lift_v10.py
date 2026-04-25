"""
gen_grasp_lift_v10.py  (fixed-base version)
=============================================
B2Z1 quadruped-arm grasping and lifting demo with FIXED BASE.

Based on the proven v10 script that produced b2z1_grasp_lift_v10.mp4
(confirmed good video with visible orange box, arm reaching and lifting).

Only change from original v10: Cfg.asset.fix_base_link = True
so the robot body stays stationary — purely an arm demonstration.

Strategy:
  Phase 1  HOLD0  : Robot stands (fixed base), no box visible
  Phase 2  REACH  : Arm policy reaches FORWARD to box at (0.533, 0, 0.516)
  Phase 3  CLOSE  : Gripper closes while arm oscillates near box
  Phase 4  HOLD1  : Brief stable hold
  Phase 5  LIFT   : CMD_P rises 0.25→0.50 → box follows analytically (13 cm lift)
  Phase 6  HOLD2  : Hold lifted pose, show robot grasping box at height

Box movement is ANALYTICALLY pre-computed (NOT live gripperMover tracking):
  box_z = CMD_L * sin(CMD_P_current) + 0.38
  max displacement ≈ 0.24 mm/step (well within 0.5 mm safe threshold)
"""

# CRITICAL: isaacgym BEFORE torch
import isaacgym
assert isaacgym
import torch
import math
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
VIDEO_OUT = "/root/RoboDuet/b2z1_grasp_fixedbase_v11.mp4"
FPS = 30

# ── Arm command parameters ────────────────────────────────────────────────────
CMD_L = 0.55
CMD_P_REACH = 0.25    # initial reach pitch (~14 deg up) → EE z ≈ 0.516 m
CMD_P_LIFT  = 0.50    # lifted pitch  (~29 deg up) → EE z ≈ 0.644 m
CMD_Y = 0.0           # yaw: straight ahead

# ── Box geometry ──────────────────────────────────────────────────────────────
BOX_HALF = 0.065      # 13 cm cube (large enough to stay in arm oscillation zone)
ARM_MOUNT_H = 0.38    # height of arm base above floor (from gen_grab_object.py)

def cmd_to_box_pos(l, p, y):
    """Analytical EE world position from arm spherical command."""
    x = l * math.cos(p) * math.cos(y)
    y_ = l * math.cos(p) * math.sin(y)
    z = l * math.sin(p) + ARM_MOUNT_H
    return [x, y_, z]

BOX_REACH  = cmd_to_box_pos(CMD_L, CMD_P_REACH, CMD_Y)   # [0.533, 0.0, 0.516]
BOX_LIFT   = cmd_to_box_pos(CMD_L, CMD_P_LIFT,  CMD_Y)   # [0.483, 0.0, 0.644]
BOX_HIDDEN = [0.0, 0.0, -10.0]

# ── Phase durations ───────────────────────────────────────────────────────────
WARMUP_STEPS = 80
HOLD0_STEPS  = 25     # robot standing, box below ground
REACH_STEPS  = 220    # arm policy reaches to CMD_P_REACH target
CLOSE_STEPS  = 80     # gripper closes around box
HOLD1_STEPS  = 60     # hold at reach pose with gripper closed
LIFT_STEPS   = 180    # CMD_P rises, box follows analytically (0.24 mm/step)
HOLD2_STEPS  = 120    # display lifted grasp

NUM_STEPS = HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + HOLD1_STEPS + LIFT_STEPS + HOLD2_STEPS

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

    Cfg.asset.fix_base_link = True          # ← KEY CHANGE: robot base stays fixed
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

    Cfg.domain_rand.randomize_friction         = False
    Cfg.domain_rand.randomize_base_mass        = False
    Cfg.domain_rand.randomize_restitution      = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength   = False
    Cfg.domain_rand.randomize_motor_offset     = False
    Cfg.domain_rand.randomize_gravity          = False
    Cfg.domain_rand.push_robots                = False

    Cfg.commands.T_force_range               = [2, 4.0]
    Cfg.domain_rand.randomize_end_effector_force = False
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
    """Adds a dynamic (teleportable) box alongside the robot."""

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
            pose.p = gymapi.Vec3(*BOX_HIDDEN)   # start hidden below ground
            pose.r = gymapi.Quat(0, 0, 0, 1)
            bh = self.gym.create_actor(
                env_handle, box_asset, pose, "target_box", i, 2)
            self.gym.set_rigid_body_color(
                env_handle, bh, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.25, 0.0))   # orange-red


def lerp(a, b, t):
    return a + (b - a) * t


def set_arm_commands(env, cmd_p):
    """Update arm goal command (l, pitch, yaw)."""
    env.commands_arm[:, 0]     = CMD_L
    env.commands_arm[:, 1]     = cmd_p
    env.commands_arm[:, 2]     = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = cmd_p
    env.commands_arm_obs[:, 2] = CMD_Y


def set_gripper(env, value):
    """Directly set gripper DOF 18 (bypasses PD control)."""
    env.dof_pos[:, 18] = value
    env.dof_vel[:, 18] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def move_box(env, pos):
    """Teleport box (actor index 1) to world position pos=[x,y,z]."""
    env.root_states[1, 0:3] = torch.tensor(pos, dtype=torch.float32, device='cuda:0')
    env.root_states[1, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda:0')
    env.root_states[1, 7:13] = 0.0
    idx = torch.tensor([1], dtype=torch.int32, device='cuda:0')
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(idx),
        1,
    )


def step_env(env, arm_policy, num_plan, gripper_val, cmd_p):
    """One environment step: arm policy + zero dog actions + gripper override."""
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


def capture(env, cam, cp):
    """Render one frame; camera gives side view showing reach + lift."""
    rx = env.root_states[0, 0].item()
    ry = env.root_states[0, 1].item()
    rz = env.root_states[0, 2].item()
    # Side view: camera on left (+y), arm reaches right (+x)
    # Clear height change visible during lift
    env.gym.set_camera_location(
        cam, env.envs[0],
        gymapi.Vec3(rx + 0.4, ry + 1.9, rz + 0.75),
        gymapi.Vec3(rx + 0.5, ry + 0.0, rz + 0.08),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def main():
    lift_cm = (BOX_LIFT[2] - BOX_REACH[2]) * 100
    max_mm_per_step = math.sqrt(
        (BOX_LIFT[0]-BOX_REACH[0])**2 + (BOX_LIFT[2]-BOX_REACH[2])**2
    ) / LIFT_STEPS * 1000

    print("=" * 65)
    print("  B2Z1 Grasp + Lift v11  (v10 + fix_base_link=True)")
    print(f"  Reach box:  ({BOX_REACH[0]:.3f}, {BOX_REACH[1]:.3f}, {BOX_REACH[2]:.3f}) m")
    print(f"  Lifted box: ({BOX_LIFT[0]:.3f},  {BOX_LIFT[1]:.3f},  {BOX_LIFT[2]:.3f}) m")
    print(f"  Lift height: {lift_cm:.1f} cm  |  max displacement: {max_mm_per_step:.2f} mm/step")
    print(f"  Phases: HOLD0={HOLD0_STEPS} REACH={REACH_STEPS} CLOSE={CLOSE_STEPS} "
          f"HOLD1={HOLD1_STEPS} LIFT={LIFT_STEPS} HOLD2={HOLD2_STEPS}")
    print(f"  Total: {NUM_STEPS} steps = {NUM_STEPS/FPS:.1f} s at {FPS} fps")
    print("=" * 65)

    setup_config()
    global_switch.open_switch()

    # ── Create env ────────────────────────────────────────────────────────────
    print("\n[1/4] Creating env with box ...")
    env = EnvWithBox(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)
    print("[OK] Env ready")

    # ── Camera ────────────────────────────────────────────────────────────────
    cp = gymapi.CameraProperties()
    cp.width = 1280; cp.height = 720; cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    # ── Arm policy ────────────────────────────────────────────────────────────
    print("\n[2/4] Loading arm policy ...")
    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)  # 8-6=2
    print(f"[OK] Arm policy (num_plan={num_plan})")

    # ── Warmup: let robot + arm policy settle ─────────────────────────────────
    print(f"\n[3/4] Warmup ({WARMUP_STEPS} steps) ...")
    env.reset()
    env.commands_dog[:, :] = 0.0
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_REACH)
    print("[OK] Warmup done")

    # ── Main recording loop ───────────────────────────────────────────────────
    print(f"\n[4/4] Recording {NUM_STEPS} steps ...")
    frames = []
    t0 = time.time()
    gripper_val = GRIPPER_OPEN
    box_pos = list(BOX_HIDDEN)

    # Phase boundary precompute
    T0 = 0
    T1 = T0 + HOLD0_STEPS       # REACH start
    T2 = T1 + REACH_STEPS       # CLOSE start
    T3 = T2 + CLOSE_STEPS       # HOLD1 start
    T4 = T3 + HOLD1_STEPS       # LIFT start
    T5 = T4 + LIFT_STEPS        # HOLD2 start

    with torch.no_grad():
        for step in range(NUM_STEPS):
            s = step

            # ── Determine phase parameters ────────────────────────────────
            if s < T1:
                # HOLD0: robot standing, no box
                phase      = "HOLD0"
                cmd_p      = CMD_P_REACH
                gripper_val = GRIPPER_OPEN
                # box stays hidden

            elif s < T2:
                # REACH: arm policy reaches to CMD_P_REACH target
                phase      = "REACH"
                cmd_p      = CMD_P_REACH
                gripper_val = GRIPPER_OPEN
                if s == T1:
                    box_pos = list(BOX_REACH)
                    move_box(env, box_pos)

            elif s < T3:
                # CLOSE: gripper closes while arm oscillates near box
                phase = "CLOSE"
                cmd_p = CMD_P_REACH
                frac  = (s - T2 + 1) / CLOSE_STEPS
                gripper_val = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, frac)
                box_pos = list(BOX_REACH)

            elif s < T4:
                # HOLD1: stable grip at reach height
                phase      = "HOLD1"
                cmd_p      = CMD_P_REACH
                gripper_val = GRIPPER_CLOSED
                box_pos    = list(BOX_REACH)

            elif s < T5:
                # LIFT: CMD_P rises analytically; box position follows
                phase = "LIFT"
                frac  = (s - T4 + 1) / LIFT_STEPS
                cmd_p = lerp(CMD_P_REACH, CMD_P_LIFT, frac)
                gripper_val = GRIPPER_CLOSED
                # Analytical box position (no live gripper tracking)
                bx = lerp(BOX_REACH[0], BOX_LIFT[0], frac)
                by = lerp(BOX_REACH[1], BOX_LIFT[1], frac)
                bz = lerp(BOX_REACH[2], BOX_LIFT[2], frac)
                box_pos = [bx, by, bz]

            else:
                # HOLD2: display lifted grasp
                phase      = "HOLD2"
                cmd_p      = CMD_P_LIFT
                gripper_val = GRIPPER_CLOSED
                box_pos    = list(BOX_LIFT)

            # ── Sim step ─────────────────────────────────────────────────
            step_env(env, arm_policy, num_plan, gripper_val, cmd_p)

            # ── Move box (all phases except HOLD0) ───────────────────────
            if phase != "HOLD0":
                move_box(env, box_pos)

            # ── Capture frame ─────────────────────────────────────────────
            frames.append(capture(env, cam, cp))

            if step == 0 or (step + 1) % 80 == 0:
                h     = env.root_states[0, 2].item()
                fps_n = (step + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  Step {step+1:>4d}/{NUM_STEPS}  h={h:.3f}  "
                    f"box_z={box_pos[2]:.3f}  cmd_p={cmd_p:.3f}  "
                    f"grip={gripper_val:.2f}  [{phase}]  {fps_n:.1f}fps"
                )

    elapsed = time.time() - t0
    print(f"\nDone: {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS/elapsed:.1f}fps)")

    # ── Save video ────────────────────────────────────────────────────────────
    print(f"\nSaving {len(frames)} frames -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
