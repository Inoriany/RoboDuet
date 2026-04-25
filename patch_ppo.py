import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# 1) Kill v4
print("Killing v4...")
stdin, stdout, stderr = ssh.exec_command('pkill -f auto_train_grasp_armonly')
time.sleep(5)
stdin, stdout, stderr = ssh.exec_command('ps aux | grep auto_train | grep -v grep')
ps = stdout.read().decode()
print(f"After kill: {'STILL RUNNING' if ps.strip() else 'KILLED'}")

# 2) Read ppo.py
ppo_path = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ppo_path}')
ppo_content = stdout.read().decode()
print(f"\nppo.py length: {len(ppo_content)} chars")

# 3) Add advantage normalization
# Find the line with surrogate loss and add advantage normalization before it
old_code = """            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio"""

new_code = """            # Normalize advantages (standard PPO practice)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio"""

if old_code in ppo_content:
    ppo_content_new = ppo_content.replace(old_code, new_code)
    print("Successfully patched ppo.py with advantage normalization!")
else:
    print("ERROR: Could not find target code in ppo.py")
    print("Looking for surrogate loss pattern...")
    for i, line in enumerate(ppo_content.split('\n')):
        if 'surrogate' in line.lower() and 'loss' in line.lower():
            print(f"  Line {i}: {line}")
    ssh.close()
    exit(1)

# 4) Write patched ppo.py back
# Use sftp to write
sftp = ssh.open_sftp()
with sftp.file(ppo_path, 'w') as f:
    f.write(ppo_content_new)
sftp.close()
print("Wrote patched ppo.py to remote")

# 5) Verify
stdin, stdout, stderr = ssh.exec_command(f'grep -n "Normalize advantages" {ppo_path}')
print(f"Verification: {stdout.read().decode().strip()}")

ssh.close()
print("\nDone! Ready to launch v5.")
