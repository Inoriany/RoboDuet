import paramiko
import os
import time


HOST = os.environ.get("ROBODUET_SSH_HOST")
PORT = int(os.environ.get("ROBODUET_SSH_PORT", "22"))
USER = os.environ.get("ROBODUET_SSH_USER", "root")
PASS = os.environ.get("ROBODUET_SSH_PASSWORD")

RUN_DIR = "/root/RoboDuet/runs/b2z1_grasp_real_phase1_resume/dummy-2t8y3jre_seed3776"
ARM_CKPT = RUN_DIR + "/checkpoints_arm/ac_weights_last_arm.pt"
DOG_CKPT = RUN_DIR + "/checkpoints_dog/ac_weights_last_dog.pt"


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=40)

    sftp = ssh.open_sftp()
    sftp.put(r"D:\CUHK\AIMS_5790\auto_train_grasp_real.py", "/root/RoboDuet/auto_train_grasp_real.py")
    sftp.close()

    ssh.exec_command(f'mkdir -p "{RUN_DIR}/videos"')

    launcher = f"""#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python auto_train_grasp_real.py --headless --no_wandb --num_learning_iterations 300 --num_envs 256 --run_name b2z1_grasp_real_phase1_balanced --resume --arm_resume_path {ARM_CKPT} --dog_resume_path {DOG_CKPT} > grasp_real_phase1_balanced.log 2>&1
echo \"EXIT_CODE=$?\" >> grasp_real_phase1_balanced.log
"""

    sftp = ssh.open_sftp()
    with sftp.file("/root/RoboDuet/run_grasp_real_phase1_balanced.sh", "w") as f:
        f.write(launcher)
    sftp.close()

    ssh.exec_command("pkill -f auto_train_grasp_real.py || true")
    ssh.exec_command("rm -f /root/RoboDuet/grasp_real_phase1_balanced.log")
    time.sleep(1)
    ssh.exec_command("nohup bash /root/RoboDuet/run_grasp_real_phase1_balanced.sh &")
    time.sleep(3)

    stdin, stdout, stderr = ssh.exec_command("ps aux | grep auto_train_grasp_real.py | grep -v grep")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
