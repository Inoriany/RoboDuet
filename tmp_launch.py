"""Upload scripts and launch demo + training on remote server."""
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

# Find ninja
out, err = run('find /opt/miniconda3/envs/roboduet -name ninja -type f 2>/dev/null')
print(f'ninja locations: {out.strip()}')

# Check if there's a lock file preventing gymtorch rebuild
out, _ = run('ls -la /root/.cache/torch_extensions/py38_cu121/gymtorch/lock 2>/dev/null')
print(f'lock file: {out.strip()}')

# Delete the lock file if it exists (stale from killed process)
run('rm -f /root/.cache/torch_extensions/py38_cu121/gymtorch/lock')
run('rm -f /root/.cache/torch_extensions/py38_cu121/gymtorch/.lock')

# The issue: nohup doesn't have conda's PATH. We need to set PATH explicitly.
# Let's find where ninja lives
out, _ = run('/opt/miniconda3/envs/roboduet/bin/python -c "import ninja; print(ninja.BIN_DIR)"')
ninja_dir = out.strip()
print(f'ninja bin dir: {ninja_dir}')

# Get the current PATH
out, _ = run('echo $PATH')
current_path = out.strip()
print(f'current PATH: {current_path}')

# Launch demo with ninja in PATH
demo_cmd = (
    f'cd /root/RoboDuet && '
    f'export PATH={ninja_dir}:$PATH && '
    f'nohup /opt/miniconda3/envs/roboduet/bin/python gen_grasp_lift_v10.py '
    f'> run_fixedbase_v11.log 2>&1 &'
)
print(f'\nLaunching demo...')
out, err = run(demo_cmd)
print(f'  out={out.strip()} err={err.strip()}')

# Wait for demo to start before launching training (they share GPU)
time.sleep(5)

# Check if demo is running
out, _ = run('ps aux | grep gen_grasp | grep -v grep')
print(f'Demo process: {out.strip()}')

# Wait for demo to finish (it's ~600 steps, should be ~2-3 min)
print('\nWaiting for demo to finish...')
for i in range(60):  # wait up to 5 min
    time.sleep(5)
    out, _ = run('ps aux | grep gen_grasp | grep -v grep')
    if not out.strip():
        print(f'  Demo finished after {(i+1)*5}s')
        break
    if i % 6 == 0:
        log_out, _ = run('tail -3 /root/RoboDuet/run_fixedbase_v11.log')
        print(f'  [{(i+1)*5}s] {log_out.strip()[:120]}')
else:
    print('  Demo still running after 5 min, continuing anyway')

# Show demo log tail
out, _ = run('tail -10 /root/RoboDuet/run_fixedbase_v11.log')
print(f'\n=== Demo log (last 10 lines) ===\n{out}')

# Now launch training (full GPU available)
train_cmd = (
    f'cd /root/RoboDuet && '
    f'export PATH={ninja_dir}:$PATH && '
    f'nohup /opt/miniconda3/envs/roboduet/bin/python auto_train_grasp_armonly.py '
    f'--headless --num_envs 256 --num_learning_iterations 5000 '
    f'--run_name b2z1_grasp_armonly_v3 '
    f'--resume --arm_resume_path /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt '
    f'--no_wandb '
    f'> grasp_armonly_v3.log 2>&1 &'
)
print('Launching training...')
out, err = run(train_cmd)
print(f'  out={out.strip()} err={err.strip()}')

time.sleep(5)
out, _ = run('ps aux | grep auto_train | grep -v grep')
print(f'Training process: {out.strip()}')

out, _ = run('tail -5 /root/RoboDuet/grasp_armonly_v3.log')
print(f'\n=== Training log (first lines) ===\n{out}')

c.close()
print('\nDONE')
