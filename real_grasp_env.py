import isaacgym
assert isaacgym
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_conjugate

from go1_gym.envs.automatic import VelocityTrackingEasyEnv
from go1_gym.utils.global_switch import global_switch
from go1_gym.utils import quaternion_to_rpy

from real_grasp_rewards import RealGraspRewards


class RealGraspEnv(VelocityTrackingEasyEnv):
    def step(self, actions, actions_arm=None):
        if actions_arm is None:
            combined = actions
        else:
            actions_loco = actions
            if getattr(self.cfg.real_grasp, "ignore_dog_actions", False):
                actions_loco = torch.zeros_like(actions_loco)
            combined = torch.cat([actions_loco, actions_arm], dim=-1)
        return super().step(combined)

    def _prepare_reward_function(self):
        self.reward_container = RealGraspRewards(self)

        for key in list(self.pretrained_reward_scales.keys()):
            scale = self.pretrained_reward_scales[key]
            if scale == 0:
                self.pretrained_reward_scales.pop(key)
            else:
                self.pretrained_reward_scales[key] *= self.dt

        for key in list(self.hybrid_reward_scales.keys()):
            self.hybrid_reward_scales[key] *= self.dt

        for name, scale in self.pretrained_reward_scales.items():
            if name not in self.hybrid_reward_scales:
                self.hybrid_reward_scales[name] = scale

        self.reward_functions = []
        self.reward_names = []
        for name in self.hybrid_reward_scales.keys():
            if name == "termination":
                continue
            if hasattr(self.reward_container, "_reward_" + name):
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, "_reward_" + name))
                if name not in self.pretrained_reward_scales:
                    self.pretrained_reward_scales[name] = 0.0

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.hybrid_reward_scales.keys()
        }
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.hybrid_reward_scales.keys()
        }
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in list(self.hybrid_reward_scales.keys()) + [
                "lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual", "ep_timesteps"
            ]
        }
        global_switch.set_reward_scales(self.hybrid_reward_scales, self.pretrained_reward_scales)

    def _init_buffers(self):
        super()._init_buffers()
        self.grasp_close_thresh = self.cfg.real_grasp.close_thresh
        self.object_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_quat[:, 3] = 1.0

    def _curriculum_ranges(self):
        step = int(self.common_step_counter)
        if step < self.cfg.real_grasp.curr_stage1_steps:
            return (
                self.cfg.real_grasp.stage1_spawn_x,
                self.cfg.real_grasp.stage1_spawn_y,
                self.cfg.real_grasp.stage1_spawn_z,
            )
        if step < self.cfg.real_grasp.curr_stage2_steps:
            return (
                self.cfg.real_grasp.stage2_spawn_x,
                self.cfg.real_grasp.stage2_spawn_y,
                self.cfg.real_grasp.stage2_spawn_z,
            )
        return (
            self.cfg.real_grasp.spawn_x,
            self.cfg.real_grasp.spawn_y,
            self.cfg.real_grasp.spawn_z,
        )

    def _reset_object_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        x_rng, y_rng, z_rng = self._curriculum_ranges()
        # spawn ranges are OFFSETS from the robot base (x,y) and absolute height (z)
        dx = torch.empty(len(env_ids), device=self.device).uniform_(x_rng[0], x_rng[1])
        dy = torch.empty(len(env_ids), device=self.device).uniform_(y_rng[0], y_rng[1])
        z = torch.empty(len(env_ids), device=self.device).uniform_(z_rng[0], z_rng[1])
        # CRITICAL: add robot base world position so each env gets a local object
        base_pos = self.root_states[env_ids, :3]
        self.object_spawn_pos[env_ids, 0] = base_pos[:, 0] + dx
        self.object_spawn_pos[env_ids, 1] = base_pos[:, 1] + dy
        self.object_spawn_pos[env_ids, 2] = z  # absolute height (independent of base)
        self.object_quat[env_ids] = 0.0
        self.object_quat[env_ids, 3] = 1.0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_object_targets(env_ids)

    def _get_object_pose_in_ee(self):
        dxyz = self.object_spawn_pos - self.grasper_pos_world()
        self.obj_pose_in_ee[:] = quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)
        return self.obj_pose_in_ee[:]

    def _get_object_abg_in_ee(self):
        self.obj_abg_in_ee[:] = 0.0
        return self.obj_abg_in_ee[:]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        # ── CRITICAL FIX: set commands_arm to point at the object ──────────
        # The pre-trained arm policy follows commands_arm (L, P, Y spherical).
        # Without this, commands_arm is randomized by the parent and conflicts
        # with the grasp reward which wants the arm to reach object_spawn_pos.
        # Convert object world position → arm-relative spherical coordinates.
        mount_h = 0.38  # arm base height above ground (absolute)
        base_pos = self.root_states[:self.num_envs, :3]
        rel = self.object_spawn_pos.clone()
        rel[:, 0] -= base_pos[:, 0]   # relative to robot base x
        rel[:, 1] -= base_pos[:, 1]   # relative to robot base y
        rel[:, 2] -= mount_h           # relative to arm mount height
        L = torch.norm(rel, dim=1).clamp(min=0.01)
        P = torch.asin((rel[:, 2] / L).clamp(-0.99, 0.99))
        Y = torch.atan2(rel[:, 1], rel[:, 0])
        self.commands_arm[:, 0] = L
        self.commands_arm[:, 1] = P
        self.commands_arm[:, 2] = Y
        self.commands_arm_obs[:, 0] = L
        self.commands_arm_obs[:, 1] = P
        self.commands_arm_obs[:, 2] = Y

        if getattr(self.cfg.real_grasp, "script_gripper", False):
            close_mask = self.get_obj_dist() < self.cfg.real_grasp.auto_close_dist
            self.dof_pos[close_mask, 18] = self.cfg.real_grasp.closed_gripper
            self.dof_pos[~close_mask, 18] = self.cfg.real_grasp.open_gripper
            self.dof_vel[:, 18] = 0.0
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf

        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = (
                torch.mean(self.root_states[:self.num_envs, 2].unsqueeze(1) - self.measured_heights, dim=1)
                < self.cfg.rewards.terminal_body_height
            )
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        rpy = quaternion_to_rpy(self.base_quat)
        self.roll, self.pitch, self.y = rpy[:, 0], rpy[:, 1], rpy[:, 2]

        if global_switch.switch_open and self.cfg.hybrid.rewards.use_terminal_roll:
            reverse_buf1 = torch.logical_and(self.roll > self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] > 0.0)
            reverse_buf2 = torch.logical_and(self.roll < -self.cfg.hybrid.rewards.terminal_body_roll, self.commands_arm[:, 2] < 0.0)
            self.reverse_buf |= reverse_buf1 | reverse_buf2

        p_align = self.commands_arm[:, 1]
        l_align = self.commands_arm[:, 0]
        self.delta_z = l_align * torch.sin(p_align) + 0.38 - self.base_pos[:, 2]

        if global_switch.switch_open and self.cfg.hybrid.rewards.use_terminal_pitch:
            reverse_buf3 = torch.logical_and(self.pitch < -self.cfg.hybrid.rewards.terminal_body_pitch, self.delta_z < -self.cfg.hybrid.rewards.headupdown_thres)
            reverse_buf4 = torch.logical_and(self.pitch > self.cfg.hybrid.rewards.terminal_body_pitch, self.delta_z > self.cfg.hybrid.rewards.headupdown_thres)
            self.reverse_buf |= reverse_buf3 | reverse_buf4

        if global_switch.switch_open:
            time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.6
            self.reverse_buf = self.reverse_buf & time_exceed_half
            self.reset_buf |= self.reverse_buf

    def object_pos_world(self):
        return self.object_spawn_pos

    def grasper_pos_world(self):
        move = torch.tensor([0.1, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        move_world = quat_apply(self.end_effector_state[:, 3:7], move)
        return self.end_effector_state[:, 0:3] + move_world

    def get_obj_rel_pos(self):
        dxyz = self.object_spawn_pos - self.grasper_pos_world()
        return quat_apply(quat_conjugate(self.end_effector_state[:, 3:7]), dxyz)

    def get_obj_dist(self):
        return torch.norm(self.get_obj_rel_pos(), dim=1)

    def get_obj_height(self):
        return self.object_spawn_pos[:, 2]
