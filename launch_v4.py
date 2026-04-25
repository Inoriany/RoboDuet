import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# Kill all instances
print("Killing all training processes...")
stdin, stdout, stderr = ssh.exec_command('pkill -f auto_train_grasp_armonly')
time.sleep(5)

# Verify killed
stdin, stdout, stderr = ssh.exec_command('ps aux | grep auto_train | grep -v grep')
ps_out = stdout.read().decode()
print(f"After kill: {'STILL RUNNING' if ps_out.strip() else 'ALL KILLED'}")

# Clear the log
stdin, stdout, stderr = ssh.exec_command('> /root/RoboDuet/grasp_armonly_v4.log')
stdout.read()

# Launch SINGLE instance
arm_weights = "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt"
launch_cmd = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && "
    "conda activate roboduet && "
    "cd /root/RoboDuet && "
    "nohup python auto_train_grasp_armonly.py "
    "--headless --num_envs 256 --num_learning_iterations 5000 "
    "--run_name b2z1_grasp_armonly_v4 "
    f"--resume --arm_resume_path {arm_weights} "
    "--no_wandb "
    "> grasp_armonly_v4.log 2>&1 &"
)

print("\nLaunching single v4 instance...")
transport = ssh.get_transport()
channel = transport.open_session()
channel.exec_command(launch_cmd)

time.sleep(20)

# Verify single instance
stdin, stdout, stderr = ssh.exec_command('ps aux | grep "auto_train_grasp_armonly.py" | grep python | grep -v grep | grep -v bash')
ps_out = stdout.read().decode()
lines = [l for l in ps_out.strip().split('\n') if l.strip()]
print(f"\nRunning instances: {len(lines)}")
for l in lines:
    print(f"  {l.strip()}")

# Check log
stdin, stdout, stderr = ssh.exec_command('tail -5 /root/RoboDuet/grasp_armonly_v4.log')
print(f"\n=== Log tail ===\n{stdout.read().decode()}")

ssh.close()
print("Done!")
