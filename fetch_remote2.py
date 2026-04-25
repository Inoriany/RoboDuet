import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# Fetch rest of ppo.py
ppo_path = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat -n {ppo_path} | tail -30')
out = stdout.read().decode()
print("=== ppo.py tail ===")
print(out)

# Fetch arm_ac.py evaluate function
ac_path = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic/arm_ac.py'
stdin, stdout, stderr = ssh.exec_command(f'grep -n "def evaluate\\|tanh\\|critic_body\\|return value" {ac_path}')
out = stdout.read().decode()
print("=== arm_ac.py evaluate section ===")
print(out)

# Full evaluate method
stdin, stdout, stderr = ssh.exec_command(f'sed -n "170,185p" {ac_path}')
out = stdout.read().decode()
print("=== arm_ac.py lines 170-185 ===")
print(out)

ssh.close()
