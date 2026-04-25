import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# Fetch current remote ppo.py (lines around the update function)
ppo_path = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic/ppo.py'
stdin, stdout, stderr = ssh.exec_command(f'cat -n {ppo_path} | head -200')
out = stdout.read().decode()
print(out)

ssh.close()
