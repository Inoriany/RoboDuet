import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# Kill current training
print("Killing training...")
stdin, stdout, stderr = ssh.exec_command('pkill -f auto_train_grasp_armonly')
time.sleep(5)

# Read current ppo.py
ppo_path = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ppo_path}')
ppo_content = stdout.read().decode()

# Add return clipping before the value loss computation
# Current code (with our previous advantage normalization patch):
old_val_loss = """            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \\
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()"""

new_val_loss = """            # Clip returns to match value function output range (tanh * 30)
            returns_batch = returns_batch.clamp(-30.0, 30.0)
            target_values_batch = target_values_batch.clamp(-30.0, 30.0)

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \\
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()"""

if old_val_loss in ppo_content:
    ppo_content_new = ppo_content.replace(old_val_loss, new_val_loss)
    sftp = ssh.open_sftp()
    with sftp.file(ppo_path, 'w') as f:
        f.write(ppo_content_new)
    sftp.close()
    print("Patched ppo.py: added return clipping to [-30, 30]")
else:
    print("ERROR: Could not find value loss code!")
    # Debug
    for i, line in enumerate(ppo_content.split('\n')):
        if 'value_losses' in line.lower() or 'Value function loss' in line:
            print(f"  Line {i}: {line}")

# Verify patches
stdin, stdout, stderr = ssh.exec_command(f'grep -n "clamp\\|Normalize advantages" {ppo_path}')
print(f"\nVerification:\n{stdout.read().decode()}")

# Clear log and relaunch
stdin, stdout, stderr = ssh.exec_command('> /root/RoboDuet/grasp_armonly_v5.log')
stdout.read()

arm_weights = "/root/RoboDuet/runs/b2z1_training_v1_rtx4090/2026-03-25/auto_train/191158.951328_seed5953/checkpoints_arm/ac_weights_last_arm.pt"
launch_cmd = (
    "source /opt/miniconda3/etc/profile.d/conda.sh && "
    "conda activate roboduet && "
    "cd /root/RoboDuet && "
    "nohup python auto_train_grasp_armonly.py "
    "--headless --num_envs 256 --num_learning_iterations 5000 "
    "--run_name b2z1_grasp_armonly_v5 "
    f"--resume --arm_resume_path {arm_weights} "
    "--no_wandb "
    "> grasp_armonly_v5.log 2>&1 &"
)

print("\nLaunching v5 (tanh bound + return clipping + adv norm + critic reset)...")
transport = ssh.get_transport()
channel = transport.open_session()
channel.exec_command(launch_cmd)

time.sleep(30)

# Verify running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep "auto_train_grasp_armonly.py" | grep python | grep -v grep | grep -v bash')
ps = stdout.read().decode()
lines = [l for l in ps.strip().split('\n') if l.strip()]
print(f"\nRunning instances: {len(lines)}")

# Early results
stdin, stdout, stderr = ssh.exec_command('grep -E "close_bonus|Value function loss|Learning iteration" /root/RoboDuet/grasp_armonly_v5.log')
log = stdout.read().decode()
print(f"\n=== Early results ===\n{log}")

ssh.close()
print("Done!")
