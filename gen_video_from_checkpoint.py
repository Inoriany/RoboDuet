#!/usr/bin/env python3
"""
Generate B2Z1 simulation video from existing checkpoint
不需要重新启动训练，直接从保存的模型生成视频
在单独的SSH session中运行
"""

import sys
import os
sys.path.insert(0, '/root/RoboDuet')

# IMPORTANT: Import isaacgym BEFORE torch
import isaacgym

import torch
import numpy as np
from pathlib import Path
import time

def find_latest_checkpoint():
    """Find the most recent checkpoint from training"""
    runs_dir = Path('/root/RoboDuet/runs')
    
    # Look for .pt files (sorted by modification time)
    checkpoints = list(runs_dir.glob('**/ac_weights_*.pt'))
    
    if not checkpoints:
        return None
    
    # Get latest by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest

def generate_video():
    print("=" * 70)
    print("B2Z1 VIDEO GENERATOR - From Existing Checkpoint")
    print("=" * 70)
    
    try:
        # Imports
        print("\n[1/4] Importing libraries...")
        from go1_gym.envs.automatic.legged_robot_config import Cfg
        from go1_gym.envs.go1.go1_config import config_go1
        from go1_gym.envs.go1.asset_config import config_asset
        from go1_gym.envs.automatic import VelocityTrackingEasyEnv
        from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
        
        print("      [OK]")
        
        # Find checkpoint
        print("\n[2/4] Finding checkpoint...")
        checkpoint_path = find_latest_checkpoint()
        
        if not checkpoint_path:
            print("      [ERROR] No checkpoint found!")
            print("      Looking in: /root/RoboDuet/runs/")
            return False
        
        print(f"      Found: {checkpoint_path.name}")
        print(f"      Modified: {time.ctime(checkpoint_path.stat().st_mtime)}")
        print(f"      Size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Setup environment
        print("\n[3/4] Setting up environment...")
        
        cfg = Cfg()
        config_go1(cfg)
        config_asset(cfg)
        
        # Add missing env attributes
        cfg.env.keep_arm_fixed = False
        
        # Disable ALL domain randomization for inference
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.randomize_restitution = False
        cfg.domain_rand.randomize_base_mass = False
        cfg.domain_rand.randomize_com_displacement = False
        cfg.domain_rand.randomize_motor_strength = False
        cfg.domain_rand.randomize_Kp_factor = False
        cfg.domain_rand.randomize_Kd_factor = False
        cfg.domain_rand.randomize_gravity = False
        cfg.domain_rand.randomize_lag_timesteps = False
        cfg.domain_rand.randomize_rigids_after_start = False
        cfg.domain_rand.randomize_end_effector_force = False
        
        # Enable rendering (IsaacGym will handle video recording)
        cfg.viewer.render = True
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"      Device: {device}")
        
        # Create env with rendering
        env = VelocityTrackingEasyEnv(
            sim_device=device,
            headless=False,  # IMPORTANT: headless=False to enable rendering
            cfg=cfg
        )
        
        print(f"      Obs shape: {cfg.env.num_observations}")
        print(f"      Action shape: {cfg.env.num_actions}")
        
        # Load checkpoint
        print("\n[4/4] Running simulation...")
        
        # The checkpoint is specifically for the DOG (legs only) policy
        # Use dog-specific configuration, not the full B2Z1 config
        num_obs = 1
        num_obs_history = 1680
        num_privileged_obs = 2
        num_actions = 12
        
        print(f"      Creating DOG policy with:")
        print(f"        num_obs: {num_obs}")
        print(f"        num_obs_history: {num_obs_history}")
        print(f"        num_privileged_obs: {num_privileged_obs}")
        print(f"        num_actions: {num_actions}")
        
        policy = DogActorCritic(
            num_obs=num_obs,
            num_privileged_obs=num_privileged_obs,
            num_obs_history=num_obs_history,
            num_actions=num_actions,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint)
        policy = policy.to(device)
        policy.eval()
        
        print(f"      Policy loaded")
        print(f"      Rendering 600 steps (10 seconds @60fps)...")
        
        # Initialize observation history with zeros
        obs_history = torch.zeros(1, num_obs_history, device=device, dtype=torch.float32)
        
        # The environment expects 19 actions: 12 (dog/legs) + 6 (arm) + 1 (gripper)
        # But our policy only outputs 12 (dog only)
        # We need to pad with zeros for arm and gripper
        num_env_actions = 19  # Total actions expected by environment
        
        for step in range(600):
            with torch.no_grad():
                # act_inference returns [batch_size, 12] (dog actions only)
                dog_action = policy.act_inference({"obs_history": obs_history})
                
                # Pad with zeros for arm (6 actions) + gripper (1 action)
                action = torch.cat([
                    dog_action,  # [1, 12]
                    torch.zeros(1, num_env_actions - num_actions, device=device, dtype=torch.float32)  # [1, 7]
                ], dim=1)  # [1, 19]
            
            # Step the environment
            env.step(action)
            
            if step % 120 == 0:
                progress = step * 100 // 600
                print(f"        {progress}% complete")
        
        env.close()
        
        print("\n[OK] Simulation complete!")
        print("\nVideo should be saved at:")
        print("  ~/.local/share/isaacgym/")
        print("  or ~/Videos/")
        print("\nTo find it:")
        print("  find ~ -name '*.mp4' -mmin -5")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_video()
    sys.exit(0 if success else 1)
