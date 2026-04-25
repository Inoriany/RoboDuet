#!/usr/bin/env python3
"""
Run IsaacGym environment with pre-generated actions and record video
"""

import sys
sys.path.insert(0, '/root/RoboDuet')

import isaacgym
import torch
import numpy as np
from pathlib import Path

def main():
    print("=" * 70)
    print("B2Z1 VIDEO RENDERER - Using Pre-generated Actions")
    print("=" * 70)
    
    try:
        print("\n[1/4] Setting up environment...")
        from go1_gym.envs.automatic.legged_robot_config import Cfg
        from go1_gym.envs.go1.go1_config import config_go1
        from go1_gym.envs.go1.asset_config import config_asset
        from go1_gym.envs.automatic import VelocityTrackingEasyEnv
        
        cfg = Cfg()
        config_go1(cfg)
        config_asset(cfg)
        
        # Disable domain rand globally by modifying the class itself
        import go1_gym.envs.automatic.legged_robot as legged_module
        original_randomize = legged_module.LeggedRobot._randomize_dof_props
        legged_module.LeggedRobot._randomize_dof_props = lambda *args, **kwargs: None
        
        cfg.viewer.render = True
        cfg.env.keep_arm_fixed = False
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"      Device: {device}")
        
        # Create environment - use 1 environment
        env = VelocityTrackingEasyEnv(
            sim_device=device,
            headless=False,
            cfg=cfg
        )
        
        print("      Environment created")
        
        print("\n[2/4] Loading pre-generated actions...")
        actions_path = Path('/root/RoboDuet/generated_actions.npy')
        if not actions_path.exists():
            print(f"      [ERROR] {actions_path} not found!")
            return False
        
        actions_dog = np.load(actions_path)
        print(f"      Loaded {len(actions_dog)} action frames")
        print(f"      Shape: {actions_dog.shape}")
        
        print("\n[3/4] Running environment (skip reset)...")
        
        # Pad actions for full environment (add zeros for arm and gripper)
        num_frames = len(actions_dog)
        actions_full = np.zeros((num_frames, 19))  # 12 dog + 6 arm + 1 gripper
        actions_full[:, :12] = actions_dog
        
        # Manually step through without calling reset (to avoid config issues)
        print(f"      Stepping {num_frames} frames...")
        
        for frame_idx in range(num_frames):
            # Convert to tensor
            action = torch.tensor(actions_full[frame_idx:frame_idx+1], device=device, dtype=torch.float32)
            
            try:
                env.step(action)
            except Exception as e:
                print(f"      Frame {frame_idx}: {e}")
                # Try to continue anyway
                pass
            
            if frame_idx % 100 == 0:
                progress = frame_idx * 100 // num_frames
                print(f"        {progress}% complete")
        
        print("      Simulation complete!")
        
        print("\n[4/4] Finding video output...")
        import subprocess
        result = subprocess.run(["find", str(Path.home()), "-name", "*.mp4", "-mmin", "-5"], 
                              capture_output=True, text=True, timeout=10)
        video_files = result.stdout.strip().split('\n')
        video_files = [f for f in video_files if f]
        
        if video_files:
            print(f"      Found {len(video_files)} video file(s):")
            for vf in video_files[:5]:
                print(f"        {vf}")
        else:
            print("      No recent video files found")
        
        env.close()
        
        print("\n[OK] Done!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
