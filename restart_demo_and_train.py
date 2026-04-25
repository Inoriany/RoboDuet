import paramiko
import time


HOST = "jq1.9gpu.com"
PORT = 11360
USER = "root"
PASS = "QBCoP-ep"

DEMO_REMOTE = "/root/RoboDuet/gen_grasp_fixedbase_success.py"
TRAIN_REMOTE = "/root/RoboDuet/auto_train_grasp_armonly.py"
ENV_REMOTE = "/root/RoboDuet/real_grasp_env.py"
REW_REMOTE = "/root/RoboDuet/real_grasp_rewards.py"

RESUME_RUN = "/root/RoboDuet/runs/b2z1_grasp_armonly_overnight/dummy-waz6pb1q_seed6989/checkpoints_arm/ac_weights_last_arm.pt"


def run(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    return out, err


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=40)

    sftp = ssh.open_sftp()
    uploads = [
        (r"D:\CUHK\AIMS_5790\gen_grasp_fixedbase_success.py", DEMO_REMOTE),
        (r"D:\CUHK\AIMS_5790\auto_train_grasp_armonly.py", TRAIN_REMOTE),
        (r"D:\CUHK\AIMS_5790\real_grasp_env.py", ENV_REMOTE),
        (r"D:\CUHK\AIMS_5790\real_grasp_rewards.py", REW_REMOTE),
    ]
    for local, remote in uploads:
        print(f"Uploading {local} -> {remote}")
        sftp.put(local, remote)

    demo_launcher = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python gen_grasp_fixedbase_success.py > run_fixedbase_success.log 2>&1
echo \"EXIT_CODE=$?\" >> run_fixedbase_success.log
"""

    train_launcher = """#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate roboduet
cd /root/RoboDuet
python auto_train_grasp_armonly.py --headless --no_wandb --num_learning_iterations 2000 --num_envs 256 --run_name b2z1_grasp_armonly_tuned --resume --arm_resume_path """ + RESUME_RUN + """ > grasp_armonly_tuned.log 2>&1
echo \"EXIT_CODE=$?\" >> grasp_armonly_tuned.log
"""

    with sftp.file("/root/RoboDuet/run_fixedbase_success.sh", "w") as f:
        f.write(demo_launcher)
    with sftp.file("/root/RoboDuet/run_grasp_armonly_tuned.sh", "w") as f:
        f.write(train_launcher)
    sftp.close()

    for cmd in [
        "pkill -f gen_grasp_fixedbase_success.py || true",
        "rm -f /root/RoboDuet/run_fixedbase_success.log",
        "nohup bash /root/RoboDuet/run_fixedbase_success.sh &",
        "pkill -f auto_train_grasp_armonly.py || true",
        "rm -f /root/RoboDuet/grasp_armonly_tuned.log",
        "nohup bash /root/RoboDuet/run_grasp_armonly_tuned.sh &",
    ]:
        run(ssh, cmd)
        time.sleep(1)

    time.sleep(3)
    out, err = run(ssh, "ps aux | grep -E 'gen_grasp_fixedbase_success.py|auto_train_grasp_armonly.py' | grep -v grep || true")
    print(out)
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
