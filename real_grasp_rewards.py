import torch

from go1_gym.envs.rewards.rewards import Rewards


class RealGraspRewards(Rewards):
    def __init__(self, env):
        super().__init__(env)

    def _reward_grasp_obj_dist(self):
        dist = self.env.get_obj_dist()
        return 1.0 / (1.0 + 6.0 * dist * dist)

    def _reward_grasp_xy_align(self):
        # WORLD-FRAME: prevents EE-rotation gaming (v8 fix)
        d = self.env.object_spawn_pos - self.env.grasper_pos_world()
        xy_sq = d[:, 0] ** 2 + d[:, 1] ** 2
        return 1.0 / (1.0 + 20.0 * xy_sq)

    def _reward_grasp_z_align(self):
        # WORLD-FRAME: prevents EE-rotation gaming (v8 fix)
        d = self.env.object_spawn_pos - self.env.grasper_pos_world()
        z = torch.abs(d[:, 2])
        return 1.0 / (1.0 + 25.0 * z * z)

    def _reward_grasp_close_bonus(self):
        # Continuous exponential proximity bonus (replaces binary threshold).
        # Provides smooth gradient at ALL distances so the policy always
        # knows which direction improves the reward.
        dist = self.env.get_obj_dist()            # norm, rotation-invariant, OK
        proximity = torch.exp(-8.0 * dist)        # broad approach signal

        # WORLD-FRAME pinch: prevents EE-rotation gaming (v8 fix)
        d = self.env.object_spawn_pos - self.env.grasper_pos_world()
        xy_dist = torch.norm(d[:, :2], dim=1)
        z_dist = torch.abs(d[:, 2])
        pinch = torch.exp(-15.0 * xy_dist) * torch.exp(-15.0 * z_dist)  # tight precision

        return proximity + 0.5 * pinch

    def _reward_grasp_lift_height(self):
        lift = (self.env.object_pos_world()[:, 2] - self.env.object_spawn_pos[:, 2]).clip(min=0.0)
        return lift

    def _reward_grasp_drop_penalty(self):
        dropped = self.env.object_pos_world()[:, 2] < (self.env.object_spawn_pos[:, 2] - 0.02)
        return dropped.float()
