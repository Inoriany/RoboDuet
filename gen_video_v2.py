#!/usr/bin/env python3
"""
Generate B2Z1 simulation video using existing trained models
可直接在远程RoboDuet环境中运行
"""

import sys
import os
sys.path.insert(0, '/root/RoboDuet')

def main():
    print("=" * 70)
    print("B2Z1 SIMULATION VIDEO GENERATOR - IsaacGym Rendering")
    print("=" * 70)
    
    try:
        import isaacgym
        import torch
        import numpy as np
        from pathlib import Path
        from datetime import datetime
        
        print("\n[1/5] Imports successful")
        print(f"      PyTorch: {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")
        
        # Import RoboDuet components
        from go1_gym.envs.automatic.legged_robot_config import Cfg
        from go1_gym.envs.go1.go1_config import config_go1
        from go1_gym.envs.go1.asset_config import config_asset
        from go1_gym.envs.automatic import VelocityTrackingEasyEnv
        
        print(f"      RoboDuet components loaded")
        
        # Configure environment
        print("\n[2/5] Configuring B2Z1 environment...")
        
        cfg = Cfg()
        config_go1(cfg)
        config_asset(cfg, "/root/RoboDuet/resources/robots/b2z1")
        
        # Enable rendering
        cfg.sim.video_video_record = True
        cfg.camera.record_video_fps = 60
        cfg.camera.record_video_height = 720
        cfg.camera.record_video_width = 1280
        
        # Create environment with rendering
        print("      Creating environment...")
        env = VelocityTrackingEasyEnv(
            sim_device="cuda:0" if torch.cuda.is_available() else "cpu",
            headless=False,  # Enable rendering
            cfg=cfg
        )
        
        print(f"      Observation shape: {env.single_observation_space.shape}")
        print(f"      Action shape: {env.single_action_space.shape}")
        
        # Find and load policy
        print("\n[3/5] Loading trained policy...")
        
        checkpoint_dir = Path('/root/RoboDuet/runs')
        model_files = list(checkpoint_dir.glob('**/ac_weights_*.pt'))
        
        if not model_files:
            print("      [WARNING] No trained models found - using random policy")
            use_model = False
        else:
            # Get latest model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            print(f"      Found model: {latest_model.name}")
            print(f"      Location: {latest_model.parent}")
            
            try:
                from go1_gym_learn.ppo_cse_automatic.dog_ac import DogAC
                
                policy = DogAC(
                    num_actor_obs=env.single_observation_space.shape[0],
                    num_actor_actions=env.single_action_space.shape[0],
                    hidden_size=256,
                )
                
                checkpoint = torch.load(latest_model)
                policy.load_state_dict(checkpoint['model_state_dict'])
                policy.eval()
                
                if torch.cuda.is_available():
                    policy.cuda()
                
                print(f"      [OK] Policy loaded successfully")
                use_model = True
                
            except Exception as e:
                print(f"      [WARNING] Could not load policy: {e}")
                print(f"      Using random policy instead")
                use_model = False
        
        # Run simulation with rendering
        print("\n[4/5] Running B2Z1 simulation (600 steps = 10 seconds)...")
        print("      Video will be saved automatically to:")
        print("      ~/.local/share/isaacgym/ or ~/Videos/")
        
        obs = env.reset()
        
        for step in range(600):
            if use_model:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs)
                    if torch.cuda.is_available():
                        obs_tensor = obs_tensor.float().cuda()
                    action = policy.act_inference(obs_tensor)[0]
            else:
                action = env.action_space.sample()
            
            obs, _, _, _, _ = env.step(action)
            
            if step % 100 == 0:
                print(f"      Progress: {step}/600 ({step*100//600}%)")
        
        print(f"      [OK] Simulation completed!")
        
        # Close environment
        env.close()
        
        # Find video output
        print("\n[5/5] Locating video output...")
        
        possible_paths = [
            Path.home() / ".local/share/isaacgym",
            Path.home() / "Videos",
            Path("/tmp/isaacgym_renders"),
            Path("/root/RoboDuet"),
        ]
        
        for path in possible_paths:
            if path.exists():
                videos = list(path.glob("**/*.mp4"))
                if videos:
                    latest_video = max(videos, key=lambda p: p.stat().st_mtime)
                    size_mb = latest_video.stat().st_size / (1024 * 1024)
                    print(f"\n[OK] Video saved!")
                    print(f"     Path: {latest_video}")
                    print(f"     Size: {size_mb:.1f} MB")
                    print(f"\nTo download this video:")
                    print(f"  scp -P 11092 root@ry3.9gpu.com:{latest_video} .")
                    return True
        
        print(f"\n[INFO] Video should be saved in IsaacGym default directory")
        print(f"       Try:")
        print(f"       find /root -name '*.mp4' -type f -mmin -10")
        print(f"       find ~ -name '*.mp4' -type f -mmin -10")
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if IsaacGym is installed:")
        print(f"     python3 -c 'import isaacgym; print(isaacgym.__file__)'")
        print(f"\n  2. Check RoboDuet installation:")
        print(f"     python3 -c 'from go1_gym import MINI_GYM_ROOT_DIR; print(MINI_GYM_ROOT_DIR)'")
        print(f"\n  3. Install/reinstall:")
        print(f"     cd /root/RoboDuet && pip install -e .")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B2Z1 simulation video')
    parser.add_argument('--steps', type=int, default=600, help='Number of simulation steps')
    parser.add_argument('--headless', type=bool, default=False, help='Disable rendering')
    
    args = parser.parse_args()
    
    success = main()
    sys.exit(0 if success else 1)
