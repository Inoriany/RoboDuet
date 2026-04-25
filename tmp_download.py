import paramiko
import os

HOST = 'jq1.9gpu.com'
PORT = 11360
USER = 'root'
PASS = 'QBCoP-ep'
LOCAL_DIR = r'D:\CUHK\AIMS_5790'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, username=USER, password=PASS)
sftp = c.open_sftp()

# Download video
remote_path = '/root/RoboDuet/b2z1_grasp_fixedbase_v11.mp4'
local_path = os.path.join(LOCAL_DIR, 'b2z1_grasp_fixedbase_v11.mp4')
print(f'Downloading {remote_path}...')
sftp.get(remote_path, local_path)
size = os.path.getsize(local_path)
print(f'Downloaded: {size/1024:.0f} KB')

sftp.close()
c.close()

# Extract frames for verification
import subprocess
out_dir = os.path.join(LOCAL_DIR, 'tmp_video_check5')
os.makedirs(out_dir, exist_ok=True)

# Use ffmpeg to extract 5 frames
total_frames = 685
for i, frame_num in enumerate([0, 170, 340, 510, 684]):
    out_file = os.path.join(out_dir, f'frame_{i}_{frame_num}.png')
    cmd = [
        'ffmpeg', '-y', '-i', local_path,
        '-vf', f'select=eq(n\\,{frame_num})',
        '-vframes', '1',
        out_file
    ]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(out_file):
        print(f'Extracted frame {frame_num}')
    else:
        print(f'FAILED to extract frame {frame_num}')

print('Done extracting frames')
