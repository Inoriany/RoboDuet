"""
test_arm_manip.py  --  Test arm manipulation checkpoint (run after ~12h of training)

Usage:
    python scripts/test_arm_manip.py
    python scripts/test_arm_manip.py --steps 600 --out /root/RoboDuet/arm_manip_demo.mp4
"""
import isaacgym
import torch
import imageio
import math
import glob
import os
import argparse
from isaacgym import gymapi
import sys
sys.path.insert(0, '/root/RoboDuet')

import play_b2z1 as p
from go1_gym.envs.automatic import HistoryWrapper, VelocityTrackingEasyEnv
from go1_gym.utils.global_switch import global_switch
from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
from go1_gym_learn.ppo_cse_automatic.arm_ac import ArmActorCritic


def find_latest_ckpt(run_dir, kind='dog'):
    """Find the most recently modified last checkpoint under run_dir."""
    pattern = os.path.join(run_dir, f'**/checkpoints_{kind}/ac_weights_last_{kind}.pt')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def make_cam(gym, sim, envs, w=1280, h=720):
    cp = gymapi.CameraProperties()
    cp.width = w; cp.height = h; cp.enable_tensors = False
    cam = gym.create_camera_sensor(envs[0], cp)
    return cam, cp


def capture(gym, sim, envs, cam, cp, rs):
    rx, ry, rz = rs[0,0].item(), rs[0,1].item(), rs[0,2].item()
    gym.set_camera_location(cam, envs[0],
        gymapi.Vec3(rx+1.5, ry+0.8, rz+0.45),
        gymapi.Vec3(rx, ry, rz+0.1))
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = gym.get_camera_image(sim, envs[0], cam, gymapi.IMAGE_COLOR)
    return img.reshape(cp.height, cp.width, 4)[:, :, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--out', type=str, default='/root/RoboDuet/arm_manip_demo.mp4')
    parser.add_argument('--run_dir', type=str, default='/root/RoboDuet/runs/b2z1_grasp_stand')
    args = parser.parse_args()

    # ---- Find checkpoints ----
    dog_ckpt = find_latest_ckpt(args.run_dir, 'dog')
    arm_ckpt = find_latest_ckpt(args.run_dir, 'arm')
    if dog_ckpt is None or arm_ckpt is None:
        print(f'ERROR: No checkpoint found under {args.run_dir}')
        print('Make sure training has been running for at least a few hours.')
        return
    print(f'[CKPT] dog: {dog_ckpt}')
    print(f'[CKPT] arm: {arm_ckpt}')

    # ---- Config (same as grasp training) ----
    p.setup_config()
    p.Cfg.dog.control.stiffness_leg['joint'] = 200.0
    p.Cfg.dog.control.damping_leg['joint']   = 20.0
    p.Cfg.control.stiffness['joint']         = 200.0
    p.Cfg.control.damping['joint']           = 20.0
    p.Cfg.env.keep_arm_fixed = False          # arm active
    # Locomotion: stand still
    p.Cfg.commands.lin_vel_x   = [0.0, 0.0]
    p.Cfg.commands.lin_vel_y   = [0.0, 0.0]
    p.Cfg.commands.ang_vel_yaw = [0.0, 0.0]
    p.Cfg.commands.limit_vel_x    = [0.0, 0.0]
    p.Cfg.commands.limit_vel_y    = [0.0, 0.0]
    p.Cfg.commands.limit_vel_yaw  = [0.0, 0.0]
    # Disable terminal reset for testing
    p.Cfg.rewards.use_terminal_body_height = False
    p.Cfg.env.max_episode_length = 99999
    global_switch.pretrained_to_hybrid_start = 0
    global_switch.pretrained_to_hybrid_end   = 0

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, num_envs=1, cfg=p.Cfg)
    env = HistoryWrapper(env)

    # ---- Load policies ----
    dog_obs_hist = p.Cfg.dog.dog_num_obs_history
    dog_priv     = p.Cfg.dog.dog_num_privileged_obs
    arm_obs_hist = p.Cfg.arm.arm_num_obs_history
    arm_priv     = p.Cfg.arm.arm_num_privileged_obs
    num_arm_act  = p.Cfg.env.num_actions - 12

    dog_policy = DogActorCritic(1, dog_priv, dog_obs_hist, 12).to('cuda:0')
    dog_policy.load_state_dict(torch.load(dog_ckpt, map_location='cuda:0'))
    dog_policy.eval()

    arm_policy = ArmActorCritic(1, arm_priv, arm_obs_hist, num_arm_act).to('cuda:0')
    arm_policy.load_state_dict(torch.load(arm_ckpt, map_location='cuda:0'))
    arm_policy.eval()
    print('[OK] Both policies loaded')

    env.reset()
    env.commands_dog[:, 0] = 0.0
    env.commands_dog[:, 1:5] = 0.0

    # Pre-fill observation history
    dog_num_obs = p.Cfg.dog.dog_num_observations
    dog_slots   = dog_obs_hist // dog_num_obs
    dog_obs0    = env.get_dog_observations()
    env.dog_obs_history[:] = dog_obs0['obs'].repeat(1, dog_slots)

    # ---- Camera ----
    cam, cp = make_cam(env.gym, env.sim, env.envs)
    frames = []

    # ---- Warmup ----
    print('Warmup 50 steps...')
    with torch.no_grad():
        for _ in range(50):
            dog_obs = env.get_dog_observations()
            arm_obs = env.get_arm_observations()
            act_dog = dog_policy.act_teacher(dog_obs['obs_history'], dog_obs['privileged_obs'])
            act_arm = arm_policy.act_teacher(arm_obs['obs_history'], arm_obs['privileged_obs'])
            env.step(act_dog, act_arm)
            env.commands_dog[:, 0] = 0.0; env.commands_dog[:, 1:5] = 0.0

    print(f'Recording {args.steps} steps...')
    with torch.no_grad():
        for i in range(args.steps):
            dog_obs = env.get_dog_observations()
            arm_obs = env.get_arm_observations()
            act_dog = dog_policy.act_teacher(dog_obs['obs_history'], dog_obs['privileged_obs'])
            act_arm = arm_policy.act_teacher(arm_obs['obs_history'], arm_obs['privileged_obs'])
            rew_dog, rew_arm, done, _ = env.step(act_dog, act_arm)
            env.commands_dog[:, 0] = 0.0; env.commands_dog[:, 1:5] = 0.0

            h = env.root_states[0, 2].item()
            if i % 100 == 0:
                print(f'step {i+1}: h={h:.3f} rew_dog={rew_dog.item():.3f} rew_arm={rew_arm.item():.3f}', flush=True)

            frames.append(capture(env.gym, env.sim, env.envs, cam, cp, env.root_states[:1]))

    writer = imageio.get_writer(args.out, fps=30, codec='libx264', quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    print(f'[DONE] Saved {args.out}  ({len(frames)} frames = {len(frames)/30:.1f}s)')


if __name__ == '__main__':
    main()
