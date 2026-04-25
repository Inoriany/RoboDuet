"""
Launch training v6b (after critical object-position bug fix):
  - Kill current v6
  - Patch remote ppo.py: add return normalization (replace tanh bound approach)
  - Upload fixed local files
  - Launch training
"""
import paramiko
import time
import os

HOST = 'jq1.9gpu.com'
PORT = 11360
USER = 'root'
PASS = 'QBCoP-ep'
REMOTE_DIR = '/root/RoboDuet'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15)
sftp = ssh.open_sftp()


def run(cmd, label=""):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if label:
        print(f"[{label}] {out.strip()}")
    if err.strip():
        print(f"  STDERR: {err.strip()}")
    return out


# ── 1. Kill v6 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Kill v6 training")
run('pkill -f auto_train_grasp_armonly')
time.sleep(5)
ps = run('ps aux | grep auto_train | grep -v grep', 'process check')
if ps.strip():
    print("  WARNING: process still running, force killing...")
    run('pkill -9 -f auto_train_grasp_armonly')
    time.sleep(3)
else:
    print("  v6 killed successfully")

# ── 2. Patch ppo.py: add value target normalization ──────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Patch ppo.py - add value target normalization")
ppo_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ppo_path}')
ppo_content = stdout.read().decode()

# Current state after v6 patch: advantage norm is there, return clamping removed.
# We need to add return normalization before the value function loss.
old_vf_section = """            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \\
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()"""

new_vf_section = """            # Normalize returns for value function stability
            ret_mean = returns_batch.mean()
            ret_std = returns_batch.std().clamp(min=1e-4)
            returns_norm = (returns_batch - ret_mean) / ret_std
            target_values_norm = (target_values_batch - ret_mean) / ret_std
            value_norm = (value_batch - ret_mean) / ret_std

            # Value function loss (on normalized values)
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_norm + \\
                                (value_norm - target_values_norm).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_norm - returns_norm).pow(2)
                value_losses_clipped = (value_clipped - returns_norm).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_norm - value_norm).pow(2).mean()"""

if old_vf_section in ppo_content:
    ppo_new = ppo_content.replace(old_vf_section, new_vf_section)
    with sftp.file(ppo_path, 'w') as f:
        f.write(ppo_new)
    print("  Added return normalization to value function loss")
else:
    print("  ERROR: Could not find value function loss section!")
    for i, line in enumerate(ppo_content.split('\n'), 1):
        if 'value_loss' in line.lower() or 'returns_batch' in line:
            print(f"    Line {i}: {line}")

# Verify
out = run(f'grep -n "returns_norm\\|Normalize returns\\|value_norm" {ppo_path}', 'verify ppo')

# ── 3. Verify arm_ac.py is clean (no tanh) ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Verify arm_ac.py")
ac_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py'
out = run(f'grep -n "tanh.*30\\|evaluate" {ac_path}', 'verify arm_ac')

# ── 4. Upload updated local files ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Upload updated local files")
files_to_upload = [
    'auto_train_grasp_armonly.py',
    'real_grasp_rewards.py',
    'real_grasp_env.py',
]
for fname in files_to_upload:
    local = os.path.join(LOCAL_DIR, fname)
    remote = f'{REMOTE_DIR}/{fname}'
    sftp.put(local, remote)
    print(f"  Uploaded {fname}")

# ── 5. Launch v6b training ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Launch v6b training")
launch_cmd = (
    'source /opt/miniconda3/etc/profile.d/conda.sh && '
    'conda activate roboduet && '
    'cd /root/RoboDuet && '
    'nohup python auto_train_grasp_armonly.py '
    '--headless --num_envs 256 --num_learning_iterations 5000 '
    '--run_name b2z1_grasp_armonly_v6 '
    '--resume --arm_resume_path /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt '
    '--no_wandb > grasp_armonly_v6.log 2>&1 &'
)
ssh.exec_command(f'bash -c "{launch_cmd}"')
time.sleep(10)

# Verify launch
ps = run('ps aux | grep auto_train | grep -v grep', 'process check')
if ps.strip():
    print("  v6b training launched!")
else:
    print("  WARNING: process not found! Checking log...")

time.sleep(15)
out = run('tail -15 /root/RoboDuet/grasp_armonly_v6.log', 'initial log')

# Check diagnostics
time.sleep(5)
out = run('grep "DIAG" /root/RoboDuet/grasp_armonly_v6.log', 'diagnostics')

sftp.close()
ssh.close()
print("\n" + "=" * 60)
print("DONE. Monitor with: python check_v6.py")
