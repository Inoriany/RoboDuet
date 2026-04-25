"""
Launch training v6:
  - Kill v5
  - Patch remote arm_ac.py: REMOVE tanh bound (caused gradient death in v5)
  - Patch remote ppo.py: REMOVE return clamping (paired with tanh, now unnecessary)
  - Upload updated local files (rewards, env, train script)
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


# ── 1. Kill v5 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Kill v5 training")
run('pkill -f auto_train_grasp_armonly')
time.sleep(5)
ps = run('ps aux | grep auto_train | grep -v grep', 'process check')
if ps.strip():
    print("  WARNING: process still running, force killing...")
    run('pkill -9 -f auto_train_grasp_armonly')
    time.sleep(3)
else:
    print("  v5 killed successfully")

# ── 2. Patch arm_ac.py: REMOVE tanh bound ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Patch arm_ac.py - remove tanh bound")
ac_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/arm_ac.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ac_path}')
ac_content = stdout.read().decode()

old_eval = """    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.critic_history_encoder(obs_h)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        # Bound value output to prevent bootstrap feedback explosion
        value = torch.tanh(value) * 30.0
        return value"""

new_eval = """    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.critic_history_encoder(obs_h)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        return value"""

if old_eval in ac_content:
    ac_new = ac_content.replace(old_eval, new_eval)
    with sftp.file(ac_path, 'w') as f:
        f.write(ac_new)
    print("  Removed tanh bound from evaluate()")
else:
    # Maybe already reverted? Check for the clean version
    if "torch.tanh(value)" in ac_content:
        print("  ERROR: tanh found but pattern doesn't match. Manual fix needed!")
        for i, line in enumerate(ac_content.split('\n'), 1):
            if 'tanh' in line and 'value' in line:
                print(f"    Line {i}: {line}")
    else:
        print("  tanh bound already removed (clean)")

# Verify
out = run(f'grep -n "tanh.*value\\|evaluate" {ac_path}', 'verify arm_ac')

# ── 3. Patch ppo.py: REMOVE return clamping ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Patch ppo.py - remove return clamping")
ppo_path = f'{REMOTE_DIR}/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ppo_path}')
ppo_content = stdout.read().decode()

# Remove the two clamp lines that were added for v5
old_clamp = """            # Clip returns to match value function output range (tanh * 30)
            returns_batch = returns_batch.clamp(-30.0, 30.0)
            target_values_batch = target_values_batch.clamp(-30.0, 30.0)

            # Value function loss"""

new_clamp = """            # Value function loss"""

if old_clamp in ppo_content:
    ppo_new = ppo_content.replace(old_clamp, new_clamp)
    with sftp.file(ppo_path, 'w') as f:
        f.write(ppo_new)
    print("  Removed return clamping lines")
else:
    if "clamp(-30" in ppo_content:
        print("  ERROR: clamp found but pattern doesn't match!")
        for i, line in enumerate(ppo_content.split('\n'), 1):
            if 'clamp' in line:
                print(f"    Line {i}: {line}")
    else:
        print("  Return clamping already removed (clean)")

# Keep advantage normalization (line 134-135) - that's good
out = run(f'grep -n "Normalize advantages\\|clamp\\|returns_batch" {ppo_path}', 'verify ppo')

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

# ── 5. Launch v6 training ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Launch v6 training")
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
if 'v6' in ps:
    print("  v6 training launched successfully!")
else:
    print("  WARNING: v6 process not found! Checking log...")
    run('head -20 /root/RoboDuet/grasp_armonly_v6.log', 'log head')

# Check first few lines of log after a bit more time
time.sleep(15)
out = run('tail -10 /root/RoboDuet/grasp_armonly_v6.log', 'initial log')

sftp.close()
ssh.close()
print("\n" + "=" * 60)
print("DONE. Monitor with: python check_v6.py")
