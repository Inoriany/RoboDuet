import paramiko
import os
import time


HOST = os.environ.get("ROBODUET_SSH_HOST")
PORT = int(os.environ.get("ROBODUET_SSH_PORT", "22"))
USER = os.environ.get("ROBODUET_SSH_USER", "root")
PASS = os.environ.get("ROBODUET_SSH_PASSWORD")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=40)

    sftp = ssh.open_sftp()
    for local, remote in [
        (r"D:\CUHK\AIMS_5790\auto_train_grasp_armonly.py", "/root/RoboDuet/auto_train_grasp_armonly.py"),
        (r"D:\CUHK\AIMS_5790\real_grasp_env.py", "/root/RoboDuet/real_grasp_env.py"),
        (r"D:\CUHK\AIMS_5790\real_grasp_rewards.py", "/root/RoboDuet/real_grasp_rewards.py"),
    ]:
        sftp.put(local, remote)
    sftp.close()

    launcher = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python auto_train_grasp_armonly.py --headless --no_wandb --num_learning_iterations 300 --num_envs 256 --run_name b2z1_grasp_armonly > grasp_armonly.log 2>&1
echo \"EXIT_CODE=$?\" >> grasp_armonly.log
"""

    sftp = ssh.open_sftp()
    with sftp.file("/root/RoboDuet/run_grasp_armonly.sh", "w") as f:
        f.write(launcher)
    sftp.close()

    ssh.exec_command("pkill -f auto_train_grasp_armonly.py || true")
    ssh.exec_command("rm -f /root/RoboDuet/grasp_armonly.log")
    time.sleep(1)
    ssh.exec_command("nohup bash /root/RoboDuet/run_grasp_armonly.sh &")
    time.sleep(3)

    stdin, stdout, stderr = ssh.exec_command("ps aux | grep auto_train_grasp_armonly.py | grep -v grep")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
