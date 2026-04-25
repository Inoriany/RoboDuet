#!/bin/bash
# Generate B2Z1 robot simulation video with rendering

cd /root/RoboDuet

# Find the latest model checkpoint
LATEST_MODEL=$(find /root/RoboDuet/runs -name "ac_weights_*.pt" -type f 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "[ERROR] No model found!"
    exit 1
fi

echo "[OK] Found latest model: $LATEST_MODEL"
echo "[1/2] Starting B2Z1 simulation with rendering..."

# Run the visualization script
python3 << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, '/root/RoboDuet')
os.environ['HEADLESS'] = 'False'  # Enable rendering

import torch
import numpy as np
from go1_gym.envs.go1.go1_env import Go1Env
from go1_gym.envs_learn.ppo_cse_automatic.dog_ac import DogAC

print("[1/2] Initializing environment with rendering...")

try:
    # Create environment with rendering enabled
    env = Go1Env(
        headless=False,  # Enable rendering
        render=True,
        num_envs=1,
        device_id=0
    )
    print("[OK] Environment created")
    
    # Try to load the latest model if available
    model_path = "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/034214.230755_seed513/checkpoints_arm/ac_weights_020800.pt"
    
    if os.path.exists(model_path):
        print(f"[OK] Loading model from: {model_path}")
        policy = DogAC(
            input_shape=env.single_observation_space.shape,
            output_shape=env.single_action_space.shape,
            hidden_size=256,
            device=torch.device('cuda:0')
        )
        checkpoint = torch.load(model_path, map_location='cuda:0')
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
    else:
        print("[WARNING] Model not found, using random policy")
        policy = None
    
    print("[2/2] Running 10 seconds of simulation (600 frames)...")
    
    obs, _ = env.reset()
    
    # Run simulation loop
    for step in range(600):
        if policy:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().cuda()
                action, _ = policy.act(obs_tensor)
                action = action.cpu().numpy()
        else:
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print progress
        if step % 100 == 0:
            print(f"   Frame {step}/600 ({step*100//600}%)")
    
    print("[OK] Simulation completed!")
    print("[OK] Video frames should be saved in IsaacGym renders directory")
    
    env.close()
    
    # Try to locate rendered frames
    import glob
    import subprocess
    
    render_dir = "/tmp/isaacgym_renders"
    if os.path.exists(render_dir):
        frames = glob.glob(f"{render_dir}/*.png")
        if frames:
            print(f"[OK] Found {len(frames)} rendered frames")
            print(f"[3/3] Creating MP4 video from frames...")
            
            # Use ffmpeg to create video
            output_video = "/root/RoboDuet/b2z1_simulation.mp4"
            cmd = f"ffmpeg -y -framerate 60 -pattern_type glob -i '{render_dir}/*.png' -c:v libx264 -pix_fmt yuv420p {output_video}"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if os.path.exists(output_video):
                size_mb = os.path.getsize(output_video) / (1024*1024)
                print(f"[OK] Video created: {output_video} ({size_mb:.1f} MB)")
            else:
                print("[WARNING] ffmpeg failed to create video")
    else:
        print("[WARNING] Render directory not found")
        print("[INFO] IsaacGym may have saved frames in another location")

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Video generation completed!"
echo "Check for output at:"
echo "  /root/RoboDuet/b2z1_simulation.mp4"
echo "=========================================="
