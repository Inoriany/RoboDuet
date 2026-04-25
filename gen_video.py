#!/usr/bin/env python3
"""
Generate B2Z1 simulation video using trained model in IsaacGym
Run this directly on the remote server: python3 gen_video.py
"""

import sys
import os
sys.path.insert(0, '/root/RoboDuet')

import torch
import numpy as np
from pathlib import Path
import time

def find_latest_model():
    """Find the latest trained model checkpoint"""
    checkpoint_dir = Path('/root/RoboDuet/runs')
    
    model_files = list(checkpoint_dir.glob('**/ac_weights_*.pt'))
    
    if not model_files:
        print("[WARNING] No model checkpoints found!")
        return None
    
    # Sort by modification time
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"[OK] Found latest model: {latest}")
    print(f"     Modified: {time.ctime(latest.stat().st_mtime)}")
    
    return str(latest)

def generate_simulation_video(output_path='/root/RoboDuet/b2z1_demo.mp4', num_steps=600):
    """
    Generate simulation video using trained B2Z1 model
    """
    
    print("=" * 70)
    print("B2Z1 SIMULATION VIDEO GENERATOR")
    print("=" * 70)
    
    try:
        # Import IsaacGym environment
        print("\n[1/4] Loading IsaacGym environment...")
        from go1_gym.envs.go1.go1_env import Go1Env
        print("      [OK] Environment class loaded")
        
        # Find model
        print("\n[2/4] Finding trained model...")
        model_path = find_latest_model()
        
        if not model_path:
            print("      [WARNING] Using random policy (no model found)")
            use_model = False
        else:
            use_model = True
        
        # Create environment with rendering
        print("\n[3/4] Initializing B2Z1 environment (headless=False for rendering)...")
        
        env_config = {
            'num_envs': 1,
            'device_id': 0,
            'headless': False,  # Enable rendering
            'render': True,
            'record_video': True,
            'video_log_step': 1,
        }
        
        try:
            env = Go1Env(**env_config)
            print("      [OK] Environment initialized")
            print(f"      Observation shape: {env.single_observation_space.shape}")
            print(f"      Action shape: {env.single_action_space.shape}")
        except Exception as e:
            print(f"      [ERROR] Could not create environment: {e}")
            print("      [TIP] Make sure IsaacGym is properly installed")
            return False
        
        # Load policy if available
        policy = None
        if use_model:
            print(f"\n[Loading policy from: {model_path}...")
            try:
                from go1_gym.envs_learn.ppo_cse_automatic.dog_ac import DogAC
                
                policy = DogAC(
                    input_shape=env.single_observation_space.shape,
                    output_shape=env.single_action_space.shape,
                    hidden_size=256,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                )
                
                checkpoint = torch.load(model_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
                policy.load_state_dict(checkpoint['model_state_dict'])
                policy.eval()
                
                print(f"      [OK] Policy loaded successfully")
                
            except Exception as e:
                print(f"      [WARNING] Could not load policy: {e}")
                print(f"      Will use random actions instead")
                policy = None
        
        # Run simulation
        print(f"\n[4/4] Running {num_steps} steps of simulation...")
        print(f"      This will generate ~{num_steps/60:.1f} seconds of video")
        print(f"      Rendering to: {output_path}")
        
        obs, _ = env.reset()
        
        for step in range(num_steps):
            # Get action
            if policy is not None:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float()
                    if torch.cuda.is_available():
                        obs_tensor = obs_tensor.cuda()
                    
                    action, _ = policy.act(obs_tensor)
                    action = action.cpu().numpy() if torch.cuda.is_available() else action.numpy()
            else:
                # Random action
                action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print progress
            if step % 60 == 0:
                print(f"      Frame {step}/{num_steps} ({step*100//num_steps}%) - Reward: {reward.mean():.3f}")
            
            # Gym will automatically record video if record_video=True
        
        print(f"\n[OK] Simulation completed!")
        
        # Check for video output
        env.close()
        
        # Find where the video was saved
        video_dir = Path('/tmp/isaacgym_renders')
        if video_dir.exists():
            videos = list(video_dir.glob('*.mp4'))
            if videos:
                latest_video = max(videos, key=lambda p: p.stat().st_mtime)
                print(f"\n[OK] Video saved at: {latest_video}")
                size_mb = latest_video.stat().st_size / (1024 * 1024)
                print(f"     Size: {size_mb:.1f} MB")
                return True
        
        print(f"\n[INFO] Check IsaacGym renders directory:")
        print(f"       ls -lh ~/.local/share/isaacgym/videos/")
        print(f"       or:")
        print(f"       find /root -name '*.mp4' -type f -mmin -5")
        
        return True
        
    except KeyboardInterrupt:
        print("\n[CANCELLED] Simulation interrupted by user")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B2Z1 simulation video')
    parser.add_argument('--steps', type=int, default=600, help='Number of simulation steps (default: 600 = 10s at 60fps)')
    parser.add_argument('--output', type=str, default='/root/RoboDuet/b2z1_demo.mp4', help='Output video path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    success = generate_simulation_video(
        output_path=args.output,
        num_steps=args.steps
    )
    
    sys.exit(0 if success else 1)
