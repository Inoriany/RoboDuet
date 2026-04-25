#!/usr/bin/env python3
"""
Simple video generation - just run policy without full IsaacGym env
"""

import sys
sys.path.insert(0, '/root/RoboDuet')

import isaacgym
import torch
import numpy as np
from pathlib import Path

def generate_video():
    print("=" * 70)
    print("B2Z1 VIDEO GENERATOR - Simple Policy Inference Only")
    print("=" * 70)
    
    try:
        print("\n[1/3] Importing libraries...")
        from go1_gym.envs.automatic.legged_robot_config import Cfg
        from go1_gym.envs.go1.go1_config import config_go1
        from go1_gym.envs.go1.asset_config import config_asset
        from go1_gym_learn.ppo_cse_automatic.dog_ac import DogActorCritic
        print("      [OK]")
        
        print("\n[2/3] Loading checkpoint...")
        runs_dir = Path('/root/RoboDuet/runs')
        checkpoints = list(runs_dir.glob('**/ac_weights_*.pt'))
        if not checkpoints:
            print("      [ERROR] No checkpoints found!")
            return False
        
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"      Found: {checkpoint_path.name}")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"      Device: {device}")
        
        # Create policy (same as before)
        num_obs = 1
        num_obs_history = 1680
        num_privileged_obs = 2
        num_actions = 12
        
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
        
        print("      Policy loaded")
        
        print("\n[3/3] Generating action sequence...")
        
        # Just generate actions without environment
        obs_history = torch.zeros(1, num_obs_history, device=device, dtype=torch.float32)
        actions_list = []
        
        for step in range(600):
            with torch.no_grad():
                action = policy.act_inference({"obs_history": obs_history})
                actions_list.append(action.cpu().numpy())
            
            if step % 120 == 0:
                progress = step * 100 // 600
                print(f"        {progress}% complete")
        
        actions_array = np.concatenate(actions_list, axis=0)
        print(f"      Generated {len(actions_array)} action frames")
        print(f"      Action shape: {actions_array.shape}")
        
        # Save to numpy file
        output_path = Path('/root/RoboDuet/generated_actions.npy')
        np.save(output_path, actions_array)
        print(f"\n      Saved actions to: {output_path}")
        
        print("\n[OK] Action sequence generated!")
        print("      Now use IsaacGym with these actions to render video")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_video()
    sys.exit(0 if success else 1)
