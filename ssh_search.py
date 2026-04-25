import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=30)

commands = [
    r'find /root/RoboDuet -name "*.py" | xargs grep -l "value_loss\|value_function_loss\|clip_param\|learning_rate" 2>/dev/null | head -20',
    r'find /root/RoboDuet -path "*/algorithms/*" -name "*.py" | head -20',
    r'find /root/RoboDuet -name "ppo*.py" -o -name "*ppo*.py" | head -20',
    r'cat /root/RoboDuet/auto_train_grasp_armonly.py',
]

for i, cmd in enumerate(commands, 1):
    print(f'===== COMMAND {i} =====')
    print(f'> {cmd}')
    print()
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=60)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out:
        print(out)
    if err:
        print(f'STDERR: {err}')
    print()

ssh.close()
print('Done.')
