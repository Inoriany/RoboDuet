#!/usr/bin/env python3
"""
Simple script to generate B2Z1 simulation video using IsaacGym recording
"""
import paramiko
import os
import time

HOST = "ry3.9gpu.com"
PORT = 11092
USERNAME = "root"
PASSWORD = "WftpaCCs"

def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=10)
    return ssh

def main():
    print("=" * 70)
    print("B2Z1 SIMULATION VIDEO GENERATOR (Simplified)")
    print("=" * 70)
    
    ssh = ssh_connect()
    print("[OK] Connected to remote server")
    
    # Create a simpler video generation script
    video_script = """#!/bin/bash
cd /root/RoboDuet

# Check Python environment
python3 -c "import isaacgym; print('[OK] IsaacGym is available')" || {
    echo "[ERROR] IsaacGym not found"
    exit 1
}

# Run the training script with video recording
python3 << 'END_SCRIPT'
import sys
import os

# Try to use existing training code with video recording
sys.path.insert(0, '/root/RoboDuet')

try:
    from go1_gym.envs.go1.go1_env import Go1Env
    from go1_gym.envs_learn.ppo_cse_automatic.hybrid_policy import HybridPolicySimple
    import torch
    
    print("[1/3] Initializing B2Z1 environment...")
    env_cfg_path = '/root/RoboDuet/go1_gym/envs/go1/config.yaml'
    env = Go1Env(headless=False, render=True)
    
    print("[2/3] Running simulation for 10 seconds...")
    for _ in range(600):  # 10 seconds at 60 fps
        action = env.action_space.sample()
        env.step(action)
    
    print("[3/3] Video should be saved...")
    
    env.close()
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

END_SCRIPT
"""
    
    # Write and execute script
    sftp = ssh.open_sftp()
    remote_script = "/tmp/gen_video.sh"
    with sftp.file(remote_script, 'w') as f:
        f.write(video_script)
    sftp.close()
    
    ssh.exec_command(f"chmod +x {remote_script}")
    print("[OK] Video script created")
    
    # Run it
    print("[IN PROGRESS] Generating video (10 seconds)...")
    stdin, stdout, stderr = ssh.exec_command(f"bash {remote_script}")
    
    for line in stdout:
        print("  " + line.strip())
    
    errors = stderr.read().decode()
    if errors:
        print("[WARNING] Output: " + errors)
    
    ssh.close()
    print("\n[DONE] Check remote server or try alternative method")

if __name__ == "__main__":
    main()
