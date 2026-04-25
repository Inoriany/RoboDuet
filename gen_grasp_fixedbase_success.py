import isaacgym
assert isaacgym
import math
import sys
import time

import imageio
import torch
from isaacgym import gymapi, gymtorch

sys.path.insert(0, '/root/RoboDuet')

from go1_gym.envs.automatic.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.wtw_config import config_wtw
from go1_gym.envs.go1.asset_config import config_asset
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic
from go1_gym.utils.global_switch import global_switch


ARM_CKPT = (
    '/root/RoboDuet/runs/b2z1_training_v1_rtx4090/'
    '2026-03-25/auto_train/191158.951328_seed5953/'
    'checkpoints_arm/ac_weights_last_arm.pt'
)
VIDEO_OUT = '/root/RoboDuet/b2z1_grasp_fixedbase_v10.mp4'
FPS = 30

CMD_L = 0.55
CMD_P_HIGH = 0.50
CMD_P_LOW = 0.25
CMD_Y = 0.0
# Height of arm mount ABOVE the robot base link (not above world origin)
ARM_MOUNT_OFFSET = 0.38

BOX_HALF = 0.03
PROP_HIDDEN = [0.0, 0.0, -10.0]

# gripperMover rigid body index (robot body index 25); ee_gripper_link offset computed below
GRIP_BODY_IDX = 25

WARMUP_STEPS = 80
HOLD0_STEPS = 40
REACH_STEPS = 180
CLOSE_STEPS = 70
HOLD1_STEPS = 50
LIFT_STEPS = 170
HOLD2_STEPS = 90
NUM_STEPS = HOLD0_STEPS + REACH_STEPS + CLOSE_STEPS + HOLD1_STEPS + LIFT_STEPS + HOLD2_STEPS

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -0.80


def setup_cfg():
    config_go1(Cfg)
    config_wtw(Cfg)
    config_asset(Cfg)

    kp, kd = 200.0, 20.0
    Cfg.dog.control.stiffness_leg['joint'] = kp
    Cfg.dog.control.damping_leg['joint'] = kd
    Cfg.control.stiffness['joint'] = kp
    Cfg.control.damping['joint'] = kd

    Cfg.asset.fix_base_link = True
    Cfg.env.keep_arm_fixed = False
    Cfg.commands.lin_vel_x = [0.0, 0.0]
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    Cfg.commands.limit_vel_x = [0.0, 0.0]
    Cfg.commands.limit_vel_y = [0.0, 0.0]
    Cfg.commands.limit_vel_yaw = [0.0, 0.0]
    Cfg.commands.distributional_commands = False
    Cfg.commands.global_reference = False

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.domain_rand.randomize_end_effector_force = False
    Cfg.domain_rand.max_force = 15
    Cfg.domain_rand.max_force_offset = 0.01
    Cfg.control.control_type = 'M'
    Cfg.env.observe_two_prev_actions = False
    Cfg.env.num_envs = 1
    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.control.update_obs_freq = 20
    Cfg.env.num_actions = 18
    Cfg.env.num_observations = 63
    Cfg.env.priv_observe_vel = False
    Cfg.env.priv_observe_high_freq_goal = False
    Cfg.dog.dog_num_privileged_obs = 2
    Cfg.arm.arm_num_privileged_obs = 9
    Cfg.env.num_privileged_obs = 9
    Cfg.hybrid.use_vision = False
    Cfg.rewards.manip_weight_lpy = 3
    Cfg.rewards.manip_weight_rpy = 1
    Cfg.rewards.use_terminal_body_height = False
    Cfg.env.max_episode_length = 99999
    Cfg.env.episode_length_s = 99999
    Cfg.asset.render_sphere = True
    Cfg.use_rot6d = False
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/b2z1/urdf/b2z1.urdf'

    for k in [
        'randomize_friction', 'randomize_base_mass', 'randomize_restitution',
        'randomize_com_displacement', 'randomize_motor_strength',
        'randomize_motor_offset', 'randomize_gravity', 'push_robots'
    ]:
        setattr(Cfg.domain_rand, k, False)

    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end = 0
    global_switch.init_sigmoid_lr()
    global_switch.open_switch()


class DemoEnv(VelocityTrackingEasyEnv):
    def _create_envs(self):
        super()._create_envs()
        self.box_actor_indices = []

        box_opts = gymapi.AssetOptions()
        box_opts.fix_base_link = True
        box_opts.disable_gravity = True
        box_asset = self.gym.create_box(self.sim, BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2, box_opts)

        for i, env_handle in enumerate(self.envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*PROP_HIDDEN)
            pose.r = gymapi.Quat(0, 0, 0, 1)

            box_handle = self.gym.create_actor(env_handle, box_asset, pose, 'target_box', i, 2)
            self.gym.set_rigid_body_color(env_handle, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0))
            self.box_actor_indices.append(self.gym.get_actor_index(env_handle, box_handle, gymapi.DOMAIN_SIM))


def set_actor_pose(env, actor_sim_idx, pos):
    env.root_states[actor_sim_idx, 0:3] = torch.tensor(pos, dtype=torch.float32, device='cuda:0')
    env.root_states[actor_sim_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda:0')
    env.root_states[actor_sim_idx, 7:13] = 0.0
    ids = torch.tensor([actor_sim_idx], dtype=torch.int32, device='cuda:0')
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(ids),
        1,
    )


def set_gripper(env, value, frozen_leg_dof=None):
    env.dof_pos[:, 18] = value
    env.dof_vel[:, 18] = 0.0
    if frozen_leg_dof is not None:
        env.dof_pos[:, :12] = frozen_leg_dof
        env.dof_vel[:, :12] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def set_arm_commands(env, cmd_p):
    env.commands_arm[:, 0] = CMD_L
    env.commands_arm[:, 1] = cmd_p
    env.commands_arm[:, 2] = CMD_Y
    env.commands_arm_obs[:, 0] = CMD_L
    env.commands_arm_obs[:, 1] = cmd_p
    env.commands_arm_obs[:, 2] = CMD_Y


def step_env(env, arm_policy, num_plan, grip, cmd_p, frozen_leg_dof=None):
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
    set_gripper(env, grip, frozen_leg_dof)


def capture(env, cam, cp, cam_pos_w, cam_target_w):
    env.gym.set_camera_location(cam, env.envs[0], cam_pos_w, cam_target_w)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def lerp(a, b, t):
    return a + (b - a) * t


def get_ee_tip(rbs):
    """Compute ee_gripper_link world position from live gripperMover rigid body state."""
    gx_ = rbs[GRIP_BODY_IDX, 0].item()
    gy_ = rbs[GRIP_BODY_IDX, 1].item()
    gz_ = rbs[GRIP_BODY_IDX, 2].item()
    qx_ = rbs[GRIP_BODY_IDX, 3].item()
    qy_ = rbs[GRIP_BODY_IDX, 4].item()
    qz_ = rbs[GRIP_BODY_IDX, 5].item()
    qw_ = rbs[GRIP_BODY_IDX, 6].item()
    lx = (
        1 - 2*(qy_*qy_ + qz_*qz_),
        2*(qx_*qy_ + qw_*qz_),
        2*(qx_*qz_ - qw_*qy_),
    )
    EE_OFFSET = 0.086
    return (gx_ + EE_OFFSET*lx[0], gy_ + EE_OFFSET*lx[1], gz_ + EE_OFFSET*lx[2])


def main():
    torch.manual_seed(0)
    setup_cfg()

    env = DemoEnv(sim_device='cuda:0', headless=True, num_envs=1, cfg=Cfg)
    env = HistoryWrapper(env)
    box_idx = env.box_actor_indices[0]

    cp = gymapi.CameraProperties()
    cp.width = 1280
    cp.height = 720
    cp.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cp)

    arm_policy = ArmActorCritic(
        num_obs=Cfg.arm.arm_num_observations,
        num_privileged_obs=Cfg.arm.arm_num_privileged_obs,
        num_obs_history=Cfg.arm.arm_num_obs_history,
        num_actions=Cfg.arm.num_actions_arm_cd,
    ).to('cuda:0')
    arm_policy.load_state_dict(torch.load(ARM_CKPT, map_location='cuda:0'))
    arm_policy.eval()
    num_plan = Cfg.arm.num_actions_arm_cd - (Cfg.env.num_actions - 12)

    # --- Phase 1: warmup at high pitch so arm settles to resting pose ---
    print('[1/4] Reset + warmup (high pitch) ...', flush=True)
    env.reset()
    env.commands_dog[:, :] = 0.0

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_HIGH)

    # Capture leg DOF AFTER warmup — PD controller has settled the legs into
    # the best attainable standing pose; freeze at this pose for all remaining phases.
    env.gym.refresh_dof_state_tensor(env.sim)
    frozen_leg_dof = env.dof_pos[:, :12].clone()
    print(f'  frozen leg DOF (first 3): {frozen_leg_dof[0, :3].cpu().tolist()}', flush=True)

    # --- Phase 2: extra steps at low pitch so we can read actual gripper pos ---
    print('[2/4] Warmup at low pitch to locate gripper ...', flush=True)
    with torch.no_grad():
        for _ in range(60):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_LOW, frozen_leg_dof)

    # Read robot base world position
    env.gym.refresh_actor_root_state_tensor(env.sim)
    bx = env.root_states[0, 0].item()
    by = env.root_states[0, 1].item()
    bz = env.root_states[0, 2].item()

    # Read gripperMover world position AND orientation
    _rbs = env.gym.acquire_rigid_body_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    rbs = gymtorch.wrap_tensor(_rbs)
    gx = rbs[GRIP_BODY_IDX, 0].item()
    gy = rbs[GRIP_BODY_IDX, 1].item()
    gz = rbs[GRIP_BODY_IDX, 2].item()
    # Quaternion (x, y, z, w) of body 25 → compute local-X axis in world frame
    qx = rbs[GRIP_BODY_IDX, 3].item()
    qy = rbs[GRIP_BODY_IDX, 4].item()
    qz = rbs[GRIP_BODY_IDX, 5].item()
    qw = rbs[GRIP_BODY_IDX, 6].item()
    # Rotate unit X vector (1,0,0) by quaternion → local X in world frame
    local_x = (
        1 - 2*(qy*qy + qz*qz),
        2*(qx*qy + qw*qz),
        2*(qx*qz - qw*qy),
    )
    # ee_gripper_link is 0.086 m further along gripperMover local X
    EE_OFFSET = 0.086
    ee_x = gx + EE_OFFSET * local_x[0]
    ee_y = gy + EE_OFFSET * local_x[1]
    ee_z = gz + EE_OFFSET * local_x[2]
    print(f'  robot base  : ({bx:.3f}, {by:.3f}, {bz:.3f})', flush=True)
    print(f'  gripper pos : ({gx:.3f}, {gy:.3f}, {gz:.3f})', flush=True)
    print(f'  gripper quat: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})', flush=True)
    print(f'  local X axis: ({local_x[0]:.3f}, {local_x[1]:.3f}, {local_x[2]:.3f})', flush=True)
    print(f'  ee_tip pos  : ({ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f})', flush=True)

    # Place cube at actual jaw-tip (ee_gripper_link) position
    box_reach = [ee_x, ee_y, ee_z]
    print(f'  box_reach   : {[round(v, 3) for v in box_reach]}', flush=True)

    # Camera: initial target is the calibrated ee jaw tip
    cam_pos_w    = gymapi.Vec3(bx + 0.40, by + 2.0, bz + 0.75)
    cam_target_w = gymapi.Vec3(ee_x,      ee_y,      ee_z)

    # --- Phase 3: re-warmup to starting (high-pitch) pose before recording ---
    print('[3/4] Re-warmup to starting pose (high pitch) ...', flush=True)
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            step_env(env, arm_policy, num_plan, GRIPPER_OPEN, CMD_P_HIGH, frozen_leg_dof)

    # Place box visible from frame 0
    set_actor_pose(env, box_idx, box_reach)

    # Acquire live rigid body state tensor (reused every frame)
    _rbs_live = env.gym.acquire_rigid_body_state_tensor(env.sim)
    rbs_live = gymtorch.wrap_tensor(_rbs_live)

    # --- Phase 4: record the grasp sequence ---
    print('[4/4] Recording fixed-base visible grasp ...', flush=True)
    frames = []
    t0 = time.time()

    t1 = HOLD0_STEPS
    t2 = t1 + REACH_STEPS
    t3 = t2 + CLOSE_STEPS
    t4 = t3 + HOLD1_STEPS
    t5 = t4 + LIFT_STEPS

    with torch.no_grad():
        for step in range(NUM_STEPS):
            if step < t1:
                phase = 'HOLD0'
                cmd_p = CMD_P_HIGH
                grip  = GRIPPER_OPEN
            elif step < t2:
                phase = 'REACH'
                frac  = (step - t1 + 1) / REACH_STEPS
                cmd_p = lerp(CMD_P_HIGH, CMD_P_LOW, frac)
                grip  = GRIPPER_OPEN
            elif step < t3:
                phase = 'CLOSE'
                frac  = (step - t2 + 1) / CLOSE_STEPS
                cmd_p = CMD_P_LOW
                grip  = lerp(GRIPPER_OPEN, GRIPPER_CLOSED, frac)
            elif step < t4:
                phase = 'HOLD1'
                cmd_p = CMD_P_LOW
                grip  = GRIPPER_CLOSED
            elif step < t5:
                phase = 'LIFT'
                cmd_p = CMD_P_LOW
                grip  = GRIPPER_CLOSED
            else:
                phase = 'HOLD2'
                cmd_p = CMD_P_LOW
                grip  = GRIPPER_CLOSED

            step_env(env, arm_policy, num_plan, grip, cmd_p, frozen_leg_dof)

            # Refresh live rigid body state AFTER the physics step
            env.gym.refresh_rigid_body_state_tensor(env.sim)

            # Box position strategy:
            #   HOLD0, REACH, CLOSE  → box stays at calibrated reach position (on ground)
            #   HOLD1, LIFT, HOLD2   → box sticks to live ee_tip (cube "held" by gripper)
            if phase in ('HOLD1', 'LIFT', 'HOLD2'):
                curr_ee = get_ee_tip(rbs_live)
                box_pos = list(curr_ee)
            else:
                box_pos = list(box_reach)

            set_actor_pose(env, box_idx, box_pos)

            # Camera tracks box position (which equals gripper tip once held)
            cam_target_now = gymapi.Vec3(box_pos[0], box_pos[1], box_pos[2])
            cam_pos_now    = gymapi.Vec3(bx + 0.40, by + 2.0, bz + 0.75 + max(0.0, box_pos[2] - ee_z))
            frames.append(capture(env, cam, cp, cam_pos_now, cam_target_now))

            if step == 0 or (step + 1) % 80 == 0 or step == NUM_STEPS - 1:
                print(
                    f'step {step + 1}/{NUM_STEPS} phase={phase} '
                    f'cmd_p={cmd_p:.3f} grip={grip:.2f} '
                    f'box={[round(v, 3) for v in box_pos]}',
                    flush=True,
                )

    print('Saving video ...', flush=True)
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f'[DONE] saved {VIDEO_OUT} in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
