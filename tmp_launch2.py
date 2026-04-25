import paramiko
import time

HOST = 'jq1.9gpu.com'
PORT = 11360
USER = 'root'
PASS = 'QBCoP-ep'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, username=USER, password=PASS)

def run(cmd, timeout=30):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err

# Write a launcher script to remote
launcher_script = """#!/bin/bash
export PATH=/opt/miniconda3/envs/roboduet/bin:$PATH
cd /root/RoboDuet

# Run demo first
echo "Starting demo..."
/opt/miniconda3/envs/roboduet/bin/python gen_grasp_lift_v10.py > run_fixedbase_v11.log 2>&1
echo "Demo done, exit code: $?"

# Then run training
echo "Starting training..."
nohup /opt/miniconda3/envs/roboduet/bin/python auto_train_grasp_armonly.py \
    --headless --num_envs 256 --num_learning_iterations 5000 \
    --run_name b2z1_grasp_armonly_v3 \
    --resume --arm_resume_path /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt \
    --no_wandb \
    > grasp_armonly_v3.log 2>&1 &

echo "Training launched in background, PID: $!"
"""

sftp = c.open_sftp()
with sftp.open('/root/RoboDuet/launch_all.sh', 'w') as f:
    f.write(launcher_script)
sftp.close()

run('chmod +x /root/RoboDuet/launch_all.sh')

# Launch the script in background
print('Launching launcher script...')
stdin, stdout, stderr = c.exec_command(
    'nohup bash /root/RoboDuet/launch_all.sh > /root/RoboDuet/launcher.log 2>&1 &',
    timeout=10
)
try:
    stdout.read()
except:
    pass

print('Waiting for demo to start...')
time.sleep(8)

# Monitor demo progress
for i in range(90):  # up to 7.5 min
    time.sleep(5)
    out, _ = run('tail -3 /root/RoboDuet/run_fixedbase_v11.log 2>/dev/null')
    out2, _ = run('ps aux | grep gen_grasp_lift | grep -v grep')
    running = bool(out2.strip())
    
    if i % 4 == 0 or not running:
        print(f'  [{(i+1)*5}s] running={running}  {out.strip()[:150]}')
    
    if not running and i > 2:
        print(f'  Demo finished after ~{(i+1)*5}s')
        break

# Show demo result
out, _ = run('tail -15 /root/RoboDuet/run_fixedbase_v11.log')
print(f'\n=== Demo log (tail) ===\n{out}')

# Check launcher log
out, _ = run('cat /root/RoboDuet/launcher.log')
print(f'=== Launcher log ===\n{out}')

# Check if training started
time.sleep(5)
out, _ = run('ps aux | grep auto_train | grep -v grep')
print(f'Training process: {out.strip()[:200]}')

out, _ = run('tail -5 /root/RoboDuet/grasp_armonly_v3.log 2>/dev/null')
print(f'\n=== Training log ===\n{out}')

c.close()
print('\nDONE')
