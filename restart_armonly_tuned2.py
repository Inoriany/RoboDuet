import paramiko
import os
import time


HOST = os.environ.get("ROBODUET_SSH_HOST")
PORT = int(os.environ.get("ROBODUET_SSH_PORT", "22"))
USER = os.environ.get("ROBODUET_SSH_USER", "root")
PASS = os.environ.get("ROBODUET_SSH_PASSWORD")

RESUME_RUN = "/root/RoboDuet/runs/b2z1_grasp_armonly_overnight/dummy-waz6pb1q_seed6989/checkpoints_arm/ac_weights_last_arm.pt"


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=40)

    sftp = ssh.open_sftp()
    sftp.put(r"D:\CUHK\AIMS_5790\auto_train_grasp_armonly.py", "/root/RoboDuet/auto_train_grasp_armonly.py")

    launcher = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python auto_train_grasp_armonly.py --headless --no_wandb --num_learning_iterations 2000 --num_envs 256 --run_name b2z1_grasp_armonly_tuned2 --resume --arm_resume_path """ + RESUME_RUN + """ > grasp_armonly_tuned2.log 2>&1
echo \"EXIT_CODE=$?\" >> grasp_armonly_tuned2.log
"""

    with sftp.file("/root/RoboDuet/run_grasp_armonly_tuned2.sh", "w") as f:
        f.write(launcher)
    sftp.close()

    for cmd in [
        "pkill -f auto_train_grasp_armonly.py || true",
        "rm -f /root/RoboDuet/grasp_armonly_tuned2.log",
        "nohup bash /root/RoboDuet/run_grasp_armonly_tuned2.sh &",
    ]:
        ssh.exec_command(cmd)
        time.sleep(1)

    time.sleep(3)
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep auto_train_grasp_armonly.py | grep -v grep || true")
    print(stdout.read().decode(errors="ignore"))
    err = stderr.read().decode(errors="ignore")
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
