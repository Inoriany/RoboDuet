"""
gen_grasp_v12b.py
=================
Static-object grasp demo with a very small, conservative lift.

Behavior:
- object stays fully static on a pedestal before grasp
- arm moves down toward that static object
- gripper closes
- arm performs only a tiny upward lift
- box motion is precomputed from a hidden preview in the same run
"""

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


ARM_CKPT = (
    "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/"
    "2026-03-25/auto_train/191158.951328_seed5953/"
    "checkpoints_arm/ac_weights_last_arm.pt"
)
VIDEO_OUT = "/root/RoboDuet/b2z1_grasp_v12b.mp4"
FPS = 30

CMD_L = 0.55
CMD_P_LOW = 0.22
CMD_P_HIGH = 0.30
CMD_P_LIFT = 0.230
CMD_Y = 0.0

BOX_HALF = 0.025
ATTACH_OFFSET_Z = -BOX_HALF * 0.35
PROP_HIDDEN = [0.0, 0.0, -10.0]
PEDESTAL_HALF_X = 0.045
PEDESTAL_HALF_Y = 0.045
PEDESTAL_HALF_Z = 0.25

WARMUP_HIGH_STEPS = 140
PREVIEW_REACH_STEPS = 170
PREVIEW_LOW_SAMPLE_STEPS = 30
PREVIEW_LIFT_STEPS = 40
PREVIEW_LIFT_SAMPLE_STEPS = 16
RETURN_HIGH_STEPS = 110
MAX_PREVIEW_TRIES = 8
PREVIEW_MIN_BASE_H = 0.42
PREVIEW_MIN_GM_Z = 0.55
PREVIEW_MAX_GM_Z = 0.85
PREVIEW_MAX_XY_SHIFT = 0.18

HOLD0_STEPS = 55
REACH_STEPS = 180
CLOSE_STEPS = 80
HOLD1_STEPS = 35
LIFT_STEPS = 70
HOLD2_STEPS = 35

NUM_STEPS = (
    HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + HOLD1_STEPS +
    LIFT_STEPS + HOLD2_STEPS
)

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -0.80


def setup_config():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

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
    Cfg.hybrid.reward_scales.arm_action_smoothness_1 = (
        5 * Cfg.reward_scales.action_smoothness_1)
    Cfg.hybrid.reward_scales.arm_action_smoothness_2 = (
        5 * Cfg.reward_scales.action_smoothness_2)
    Cfg.use_rot6d = False
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf"
    global_switch.init_sigmoid_lr()


class EnvWithProps(VelocityTrackingEasyEnv):
    def _create_envs(self):
        super()._create_envs()

        box_opts = gymapi.AssetOptions()
        box_opts.fix_base_link = True
        box_opts.density = 500.0
        box_opts.disable_gravity = True

        pedestal_opts = gymapi.AssetOptions()
        pedestal_opts.fix_base_link = True
        pedestal_opts.density = 500.0
        pedestal_opts.disable_gravity = True

        box_asset = self.gym.create_box(
            self.sim, BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2, box_opts)
        pedestal_asset = self.gym.create_box(
            self.sim,
            PEDESTAL_HALF_X * 2,
            PEDESTAL_HALF_Y * 2,
            PEDESTAL_HALF_Z * 2,
            pedestal_opts,
        )

        for i, env_handle in enumerate(self.envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*PROP_HIDDEN)
            pose.r = gymapi.Quat(0, 0, 0, 1)

            bh = self.gym.create_actor(env_handle, box_asset, pose, "target_box", i, 2)
            self.gym.set_rigid_body_color(
                env_handle, bh, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.25, 0.0))

            ph = self.gym.create_actor(env_handle, pedestal_asset, pose, "pedestal", i, 3)
            self.gym.set_rigid_body_color(
                env_handle, ph, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.70, 0.70, 0.72))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_vec(a, b, t):
    return [lerp(a[i], b[i], t) for i in range(len(a))]


def set_arm_commands(env, cmd_p):
    env.commands_arm[:, 0] = CMD_L
    env.commands_arm[:, 1] = cmd_p
    env.commands_arm[:, 2] = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = cmd_p
    env.commands_arm_obs[:, 2] = CMD_Y


def set_gripper(env, value):
    env.dof_pos[:, 18] = value
    env.dof_vel[:, 18] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def set_actor_pose(env, actor_idx, pos):
    env.root_states[actor_idx, 0:3] = torch.tensor(pos, dtype=torch.float32, device='cuda:0')
    env.root_states[actor_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda:0')
    env.root_states[actor_idx, 7:13] = 0.0
    idx = torch.tensor([actor_idx], dtype=torch.int32, device='cuda:0')
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


def get_gripper_mover_pos(env):
    bodies = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
    return bodies[0, 25, :3].cpu().tolist()


def capture(env, cam, cp, focus_pos):
    rx = env.root_states[0, 0].item()
    ry = env.root_states[0, 1].item()
    rz = env.root_states[0, 2].item()
    env.gym.set_camera_location(
        cam, env.envs[0],
        gymapi.Vec3(rx + 0.90, ry + 1.45, rz + 0.72),
        gymapi.Vec3(focus_pos[0], focus_pos[1], focus_pos[2] + 0.02),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def dist3(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def sample_gm(env, arm_policy, num_plan, steps, cmd_p):
    gm_samples = []
    h_samples = []
    for _ in range(steps):
        step_env(env, arm_policy, num_plan, GRIPPER_OPEN, cmd_p)
        gm_samples.append(get_gripper_mover_pos(env))
        h_samples.append(env.root_states[0, 2].item())
    n = len(gm_samples)
    gm_avg = [sum(v[i] for v in gm_samples) / n for i in range(3)]
    h_avg = sum(h_samples) / n
    return gm_avg, h_avg


def run_preview_attempt(env, arm_policy, num_plan, attempt_idx):
    env.reset()
    env.commands_dog[:, :] = 0.0
    set_actor_pose(env, 1, PROP_HIDDEN)
    set_actor_pose(env, 2, PROP_HIDDEN)

    with torch.no_grad():
        for _ in range(WARMUP_HIGH_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_HIGH)

        for i in range(PREVIEW_REACH_STEPS):
            frac = (i + 1) / PREVIEW_REACH_STEPS
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, lerp(CMD_P_HIGH, CMD_P_LOW, frac))

        gm_low, h_low = sample_gm(env, arm_policy, num_plan, PREVIEW_LOW_SAMPLE_STEPS, CMD_P_LOW)

        for i in range(PREVIEW_LIFT_STEPS):
            frac = (i + 1) / PREVIEW_LIFT_STEPS
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, lerp(CMD_P_LOW, CMD_P_LIFT, frac))

        gm_lift, h_lift = sample_gm(env, arm_policy, num_plan, PREVIEW_LIFT_SAMPLE_STEPS, CMD_P_LIFT)

    xy_shift = ((gm_lift[0] - gm_low[0]) ** 2 + (gm_lift[1] - gm_low[1]) ** 2) ** 0.5
    dz = gm_lift[2] - gm_low[2]
    ok = (
        h_low >= PREVIEW_MIN_BASE_H and
        h_lift >= PREVIEW_MIN_BASE_H and
        PREVIEW_MIN_GM_Z <= gm_low[2] <= PREVIEW_MAX_GM_Z and
        PREVIEW_MIN_GM_Z <= gm_lift[2] <= PREVIEW_MAX_GM_Z and
        dz > 0.01 and
        xy_shift <= PREVIEW_MAX_XY_SHIFT
    )

    print(
        f"  Preview try {attempt_idx}: "
        f"low_h={h_low:.3f} lift_h={h_lift:.3f} "
        f"gm_low=({gm_low[0]:.3f},{gm_low[1]:.3f},{gm_low[2]:.3f}) "
        f"gm_lift=({gm_lift[0]:.3f},{gm_lift[1]:.3f},{gm_lift[2]:.3f}) "
        f"xy_shift={xy_shift:.3f} dz={dz:.3f} ok={ok}"
    )

    if not ok:
        return None

    with torch.no_grad():
        for i in range(RETURN_HIGH_STEPS):
            frac = (i + 1) / RETURN_HIGH_STEPS
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, lerp(CMD_P_LIFT, CMD_P_HIGH, frac))

    return {
        "gm_low": gm_low,
        "h_low": h_low,
        "gm_lift": gm_lift,
        "h_lift": h_lift,
        "dz": dz,
    }


def main():
    print("=" * 72)
    print("  B2Z1 Grasp v12b  (static object + tiny lift)")
    print(f"  P: {CMD_P_HIGH:.3f} -> {CMD_P_LOW:.3f} -> {CMD_P_LIFT:.3f}")
    print("  Object remains static until after close")
    print("=" * 72)

    setup_config()
    global_switch.open_switch()

    print("\n[1/4] Creating env ...")
    env = EnvWithProps(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)

    cp = gymapi.CameraProperties()
    cp.width = 1280
    cp.height = 720
    cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)
    print("[OK] Env ready")

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
    print(f"[OK] Arm policy loaded (num_plan={num_plan})")

    print("\n[3/4] Hidden preview to determine low and lift box poses ...")
    preview = None
    for attempt_idx in range(1, MAX_PREVIEW_TRIES + 1):
        preview = run_preview_attempt(env, arm_policy, num_plan, attempt_idx)
        if preview is not None:
            break
    if preview is None:
        raise RuntimeError("No stable preview found after retries")

    gm_low = preview["gm_low"]
    h_low = preview["h_low"]
    gm_lift = preview["gm_lift"]
    h_lift = preview["h_lift"]

    box_low = [gm_low[0], gm_low[1], gm_low[2] + ATTACH_OFFSET_Z]
    box_lift = [box_low[0], box_low[1], gm_lift[2] + ATTACH_OFFSET_Z]
    lift_dz = box_lift[2] - box_low[2]
    pedestal_pos = [
        box_low[0],
        box_low[1],
        max(PEDESTAL_HALF_Z, box_low[2] - BOX_HALF - PEDESTAL_HALF_Z),
    ]

    print(f"  preview low  base_h={h_low:.3f}  gm=({gm_low[0]:.3f},{gm_low[1]:.3f},{gm_low[2]:.3f})")
    print(f"  preview lift base_h={h_lift:.3f}  gm=({gm_lift[0]:.3f},{gm_lift[1]:.3f},{gm_lift[2]:.3f})")
    print(f"  box low:      ({box_low[0]:.3f}, {box_low[1]:.3f}, {box_low[2]:.3f})")
    print(f"  box lift:     ({box_lift[0]:.3f}, {box_lift[1]:.3f}, {box_lift[2]:.3f})")
    print(f"  tiny lift dz: {lift_dz * 100:.2f} cm")
    print("[OK] Preview complete")

    print(f"\n[4/4] Recording {NUM_STEPS} steps ...")
    frames = []
    t0 = time.time()

    T1 = HOLD0_STEPS
    T2 = T1 + REACH_STEPS
    T3 = T2 + CLOSE_STEPS
    T4 = T3 + HOLD1_STEPS
    T5 = T4 + LIFT_STEPS

    with torch.no_grad():
        for step in range(NUM_STEPS):
            if step < T1:
                phase = "HOLD0"
                cmd_p = CMD_P_HIGH
                gripper_val = GRIPPER_OPEN
                box_pos = list(box_low)
                pedestal_now = pedestal_pos
            elif step < T2:
                phase = "REACH"
                frac = (step - T1 + 1) / REACH_STEPS
                cmd_p = lerp(CMD_P_HIGH, CMD_P_LOW, frac)
                gripper_val = GRIPPER_OPEN
                box_pos = list(box_low)
                pedestal_now = pedestal_pos
            elif step < T3:
                phase = "CLOSE"
                frac = (step - T2 + 1) / CLOSE_STEPS
                cmd_p = CMD_P_LOW
                gripper_val = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, frac)
                box_pos = list(box_low)
                pedestal_now = pedestal_pos
            elif step < T4:
                phase = "HOLD1"
                cmd_p = CMD_P_LOW
                gripper_val = GRIPPER_CLOSED
                box_pos = list(box_low)
                pedestal_now = pedestal_pos
            elif step < T5:
                phase = "LIFT"
                frac = (step - T4 + 1) / LIFT_STEPS
                cmd_p = lerp(CMD_P_LOW, CMD_P_LIFT, frac)
                gripper_val = GRIPPER_CLOSED
                box_pos = lerp_vec(box_low, box_lift, frac)
                pedestal_now = PROP_HIDDEN
            else:
                phase = "HOLD2"
                cmd_p = CMD_P_LIFT
                gripper_val = GRIPPER_CLOSED
                box_pos = list(box_lift)
                pedestal_now = PROP_HIDDEN

            step_env(env, arm_policy, num_plan, gripper_val, cmd_p)
            gm = get_gripper_mover_pos(env)
            set_actor_pose(env, 1, box_pos)
            set_actor_pose(env, 2, pedestal_now)

            focus_pos = [
                0.7 * box_pos[0] + 0.3 * gm[0],
                0.7 * box_pos[1] + 0.3 * gm[1],
                0.7 * box_pos[2] + 0.3 * gm[2],
            ]
            frames.append(capture(env, cam, cp, focus_pos))

            if step == 0 or (step + 1) % 60 == 0 or step == NUM_STEPS - 1:
                h = env.root_states[0, 2].item()
                follow_target = [gm[0], gm[1], gm[2] + ATTACH_OFFSET_Z]
                d = dist3(box_pos, follow_target)
                fps_now = (step + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  Step {step + 1:>4d}/{NUM_STEPS}  base_h={h:.3f}  "
                    f"box=({box_pos[0]:.3f},{box_pos[2]:.3f})  "
                    f"gm=({follow_target[0]:.3f},{follow_target[2]:.3f})  "
                    f"dist={d:.3f}  cmd_p={cmd_p:.3f}  grip={gripper_val:.2f}  "
                    f"[{phase}]  {fps_now:.1f}fps"
                )

    elapsed = time.time() - t0
    final_h = env.root_states[0, 2].item()
    print(f"\nDone: {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS / elapsed:.1f}fps)")
    print(f"Final base_h={final_h:.3f}")

    print(f"Saving {len(frames)} frames -> {VIDEO_OUT}")
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f"[OK] Video saved: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
