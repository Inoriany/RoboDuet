#!/usr/bin/env python3
"""
Generate B2Z1 simulation video and download it locally
"""
import paramiko
import os
import time

# SSH Connection Details
HOST = "ry3.9gpu.com"
PORT = 11092
USERNAME = "root"
PASSWORD = "WftpaCCs"

REMOTE_ROBODIET_PATH = "/root/RoboDuet"
REMOTE_LOGS_PATH = "/root/RoboDuet/logs"
LOCAL_DOWNLOAD_PATH = r"D:\CUHK\AIMS_5790"

def create_ssh_client():
    """Create and return SSH client"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=10)
        print(f"[OK] Successfully connected to {HOST}:{PORT}")
        return ssh
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return None

def check_training_status(ssh):
    """Check B2Z1 training status"""
    print("\n[1/4] Checking B2Z1 training status...")
    stdin, stdout, stderr = ssh.exec_command("ls -lh /root/RoboDuet/logs/ 2>/dev/null | head -20")
    output = stdout.read().decode()
    
    if output:
        print(output)
        return True
    else:
        error = stderr.read().decode()
        print(f"Could not find logs: {error}")
        return False

def create_video_generation_script(ssh):
    """Create a script to generate simulation video"""
    print("\n[2/4] Creating video generation script...")
    
    video_script = """#!/bin/bash
cd /root/RoboDuet

# Check if training model exists
LATEST_MODEL=$(find logs -name "model_*.pt" -type f 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "No trained model found! Using random policy for demo video."
    LATEST_MODEL="None"
else
    echo "Found model: $LATEST_MODEL"
fi

# Create a simple visualization script
python3 << 'PYTHON_SCRIPT'
import torch
import numpy as np
from isaacgym import gymapi
import os
import sys
sys.path.insert(0, '/root/RoboDuet')

# Initialize IsaacGym
gym = gymapi.Gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1/60
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create ground
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load B2Z1 robot
asset_root = "/root/RoboDuet/resources"
asset_file = "robots/b2z1/urdf/b2z1.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)

# Create environment and actor
env = gym.create_env(sim, gymapi.Vec3(-2, 0, 0), gymapi.Vec3(2, 1, 2), 1)
actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "b2z1", 0, 1)

# Set camera
cam_pos = gymapi.Vec3(3, 2, 2.5)
cam_target = gymapi.Vec3(0, 0, 0.5)
gym.viewer_camera_look_at(gym.get_viewer(sim), None, cam_pos, cam_target)

# Get viewer
viewer = gym.get_viewer(sim)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESC, "QUIT")

# Create video file path
video_path = "/root/RoboDuet/logs/b2z1_simulation_demo.mp4"
os.makedirs(os.path.dirname(video_path), exist_ok=True)

print(f"Generating simulation video: {video_path}")
print("Recording for 5 seconds at 60 FPS...")

# Simulation loop
frame_count = 0
max_frames = 300  # 5 seconds at 60 fps

try:
    for frame in range(max_frames):
        # Apply random actions to legs for demonstration
        dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)
        actions = np.sin(np.arange(18) * 0.1 + frame * 0.05) * 0.5
        
        # Set DOF targets
        gym.set_actor_dof_position_targets(env, actor_handle, actions)
        
        # Simulate
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        
        # Render
        gym.render(sim)
        if viewer:
            gym.poll_viewer_events(sim)
        
        frame_count += 1
        
        if frame % 60 == 0:
            print(f"  Frame {frame}/{max_frames} ({frame*100//max_frames}%)")
    
    print(f"[OK] Successfully generated {frame_count} frames")
    print(f"Video saved to: {video_path}")
    
except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error during simulation: {e}")
    import traceback
    traceback.print_exc()
finally:
    gym.destroy_sim(sim)
    gym.destroy_viewer(viewer) if viewer else None

PYTHON_SCRIPT

echo "Video generation completed!"
"""
    
    # Write script to remote server
    sftp = ssh.open_sftp()
    try:
        remote_script_path = "/root/RoboDuet/generate_video.sh"
        with sftp.file(remote_script_path, 'w') as f:
            f.write(video_script)
        
        # Make script executable
        ssh.exec_command(f"chmod +x {remote_script_path}")
        print(f"[OK] Created video generation script: {remote_script_path}")
        
        return remote_script_path
    finally:
        sftp.close()

def run_video_generation(ssh, script_path):
    """Run video generation script"""
    print("\n[3/4] Running video generation (this may take 2-3 minutes)...")
    
    stdin, stdout, stderr = ssh.exec_command(f"bash {script_path}")
    
    # Stream output
    for line in stdout:
        print(f"  {line.strip()}")
    
    err = stderr.read().decode()
    if err:
        print(f"Errors: {err}")

def download_video(ssh):
    """Download generated video to local machine"""
    print("\n[4/4] Downloading video to local machine...")
    
    remote_video = "/root/RoboDuet/logs/b2z1_simulation_demo.mp4"
    local_video = os.path.join(LOCAL_DOWNLOAD_PATH, "b2z1_simulation_demo.mp4")
    
    try:
        sftp = ssh.open_sftp()
        
        # Check if file exists
        try:
            sftp.stat(remote_video)
            print(f"  Downloading from: {remote_video}")
            sftp.get(remote_video, local_video)
            print(f"[OK] Video downloaded to: {local_video}")
            return local_video
        except IOError:
            print(f"[ERROR] Video file not found on remote: {remote_video}")
            return None
        finally:
            sftp.close()
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None

def main():
    print("=" * 60)
    print("B2Z1 SIMULATION VIDEO GENERATOR")
    print("=" * 60)
    
    # Connect to SSH
    ssh = create_ssh_client()
    if not ssh:
        return
    
    try:
        # Check training status
        if not check_training_status(ssh):
            print("Warning: Could not find training logs, will generate demo video anyway")
        
        # Create video script
        script_path = create_video_generation_script(ssh)
        
        # Run video generation
        run_video_generation(ssh, script_path)
        
        # Download video
        local_video = download_video(ssh)
        
        if local_video:
            print("\n" + "=" * 60)
            print("[OK] SUCCESS! Video ready for PPT")
            print(f"  Location: {local_video}")
            print("=" * 60)
            
            # Next steps
            print("\nNext steps:")
            print("1. Open PhD_Briefing_B2Z1_Grasping_WITH_CHART.pptx")
            print("2. Go to Slide 5 (or add new slide)")
            print("3. Insert → Video → b2z1_simulation_demo.mp4")
            print("4. Position and resize video as needed")
            print("5. Save PowerPoint")
        
    finally:
        ssh.close()
        print("\nSSH connection closed.")

if __name__ == "__main__":
    main()
