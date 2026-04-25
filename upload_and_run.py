"""Upload fix8 script to server and launch it via nohup."""
import paramiko
import time

HOST = "jq1.9gpu.com"
PORT = 11360
USER = "root"
PASS = "QBCoP-ep"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}:{PORT} ...")
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
print("Connected.")

# Upload gen_grasp_v11.py
sftp = ssh.open_sftp()
print("Uploading gen_grasp_v11.py ...")
sftp.put(r"D:\CUHK\AIMS_5790\gen_grasp_v11.py", "/root/RoboDuet/gen_grasp_v11.py")
print("Upload complete.")
sftp.close()

# Write launcher script
launcher = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python gen_grasp_v11.py > run_v11.log 2>&1
echo "EXIT_CODE=$?" >> run_v11.log
"""
sftp = ssh.open_sftp()
with sftp.file("/root/RoboDuet/run_v11.sh", "w") as f:
    f.write(launcher)
sftp.close()
print("Launcher script written.")

# Kill any existing run
stdin, stdout, stderr = ssh.exec_command("pkill -f gen_grasp_v11 || true")
stdout.read()
time.sleep(1)

# Clear old log
stdin, stdout, stderr = ssh.exec_command("rm -f /root/RoboDuet/run_v11.log")
stdout.read()

# Launch via nohup
print("Launching nohup bash /root/RoboDuet/run_v11.sh & ...")
stdin, stdout, stderr = ssh.exec_command(
    "nohup bash /root/RoboDuet/run_v11.sh &"
)
time.sleep(2)

# Verify it started
stdin, stdout, stderr = ssh.exec_command("ps aux | grep gen_grasp_v11 | grep -v grep")
ps_out = stdout.read().decode()
if ps_out.strip():
    print("Process is running:")
    print(ps_out.strip())
else:
    print("WARNING: Process may not have started. Checking log...")
    stdin, stdout, stderr = ssh.exec_command("cat /root/RoboDuet/run_v11.log 2>/dev/null | head -20")
    print(stdout.read().decode())

ssh.close()
print("Done. Poll run_v11.log for progress.")
