"""
Launch training v7 — key changes from v6b:
  - LR: 1e-5 (was 3e-5) — preserve pretrained arm reaching
  - Entropy: 0.03 (was 0.01) — prevent premature collapse
  - Object closer: ~0.34m forward (was ~0.48m) — easier for crouched arm
  - Episode: 2.4s (was 1.6s) — more time for arm trajectory
  - Start from original pretrained weights (not v6b checkpoint)
  - ppo.py: keep advantage norm + return normalization from v6b
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


# ── 1. Kill v6b ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Kill v6b training")
run('pkill -f auto_train_grasp_armonly')
time.sleep(5)
ps = run('ps aux | grep auto_train | grep -v grep', 'process check')
if ps.strip():
    print("  WARNING: process still running, force killing...")
    run('pkill -9 -f auto_train_grasp_armonly')
    time.sleep(3)
else:
    print("  v6b killed successfully")

# ── 2. Verify ppo.py has return normalization (from v6b) ─────────────────────
print("\n" + "=" * 60)
print("STEP 2: Verify ppo.py has return normalization")
ppo_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py'
out = run(f'grep -c "returns_norm\\|Normalize returns" {ppo_path}', 'ppo check')
count = int(out.strip()) if out.strip().isdigit() else 0
if count >= 2:
    print("  ppo.py already has return normalization (from v6b). OK.")
else:
    print("  WARNING: ppo.py may not have return normalization!")
    print("  Patching now...")
    stdin, stdout, stderr = ssh.exec_command(f'cat {ppo_path}')
    ppo_content = stdout.read().decode()

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
        print("  Patched ppo.py with return normalization")
    else:
        print("  Could not find original section — may already be patched or different")

# ── 3. Verify arm_ac.py is clean (no tanh) ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Verify arm_ac.py has no tanh bound")
ac_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py'
out = run(f'grep -c "tanh" {ac_path}', 'arm_ac tanh check')
count = int(out.strip()) if out.strip().isdigit() else 0
if count == 0:
    print("  arm_ac.py is clean. OK.")
else:
    print(f"  WARNING: found {count} tanh references in arm_ac.py")

# ── 4. Upload updated local files ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Upload v7 files")
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

# ── 5. Launch v7 training ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Launch v7 training")
print("  Key v7 changes:")
print("    LR: 1e-5 (was 3e-5)")
print("    Entropy: 0.03 (was 0.01)")
print("    Object distance: ~0.34m (was ~0.48m)")
print("    Episode: 2.4s (was 1.6s)")
launch_cmd = (
    'source /opt/miniconda3/etc/profile.d/conda.sh && '
    'conda activate roboduet && '
    'cd /root/RoboDuet && '
    'nohup python auto_train_grasp_armonly.py '
    '--headless --num_envs 256 --num_learning_iterations 5000 '
    '--run_name b2z1_grasp_armonly_v7 '
    '--resume --arm_resume_path /root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt '
    '--no_wandb > grasp_armonly_v7.log 2>&1 &'
)
ssh.exec_command(f'bash -c "{launch_cmd}"')
time.sleep(10)

# Verify launch
ps = run('ps aux | grep auto_train | grep -v grep', 'process check')
if ps.strip():
    print("  v7 training launched!")
else:
    print("  WARNING: process not found! Checking log...")
    run('tail -20 /root/RoboDuet/grasp_armonly_v7.log', 'error log')

time.sleep(20)
out = run('tail -20 /root/RoboDuet/grasp_armonly_v7.log', 'initial log')

# Check first diagnostic
time.sleep(10)
out = run('grep "DIAG" /root/RoboDuet/grasp_armonly_v7.log', 'diagnostics')

sftp.close()
ssh.close()
print("\n" + "=" * 60)
print("DONE. Monitor with:")
print("  python check_v7.py")
print("  (or modify check_v6.py to read grasp_armonly_v7.log)")
