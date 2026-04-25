import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

print('Waiting 90 seconds for ~100 iterations...')
time.sleep(90)

# Check process
stdin, stdout, stderr = ssh.exec_command('ps aux | grep auto_train | grep -v grep')
ps = stdout.read().decode()
print('Process running:', bool(ps.strip()))

# Get diagnostics
stdin, stdout, stderr = ssh.exec_command('grep "DIAG" /root/RoboDuet/grasp_armonly_v8.log')
print('\n=== DIAGNOSTICS ===')
print(stdout.read().decode())

# Get recent metrics
cmd = 'grep -E "close_bonus|Value function loss|Learning iteration|rew_total|rew_grasp_xy_align|rew_grasp_z_align" /root/RoboDuet/grasp_armonly_v8.log'
stdin, stdout, stderr = ssh.exec_command(cmd)
output = stdout.read().decode()

lines = output.strip().split('\n')
iters, close_bonus, value_loss, total_rew, xy_align, z_align = [], [], [], [], [], []
for line in lines:
    if 'Learning iteration' in line:
        iters.append(int(line.strip().split('iteration')[1].split('/')[0].strip()))
    elif 'close_bonus' in line:
        close_bonus.append(float(line.split(':')[-1].strip()))
    elif 'Value function loss' in line:
        value_loss.append(float(line.split(':')[-1].strip()))
    elif 'rew_total' in line:
        total_rew.append(float(line.split(':')[-1].strip()))
    elif 'rew_grasp_xy_align' in line:
        xy_align.append(float(line.split(':')[-1].strip()))
    elif 'rew_grasp_z_align' in line:
        z_align.append(float(line.split(':')[-1].strip()))

n = min(len(iters), len(close_bonus))
print(f'\n=== LATEST METRICS ({n} iters parsed) ===')
if n > 5:
    for i in range(max(0, n - 10), n):
        tr = total_rew[i] if i < len(total_rew) else 0
        xy = xy_align[i] if i < len(xy_align) else 0
        za = z_align[i] if i < len(z_align) else 0
        vl = value_loss[i] if i < len(value_loss) else 0
        print(f'  iter {iters[i]:4d}: close_bonus={close_bonus[i]:.4f}  xy_align={xy:.4f}  z_align={za:.4f}  total={tr:.2f}  val_loss={vl:.2f}')

    early = close_bonus[:min(20, n)]
    late = close_bonus[max(0, n - 20):]
    print(f'\nEarly avg close_bonus (first 20): {sum(early)/len(early):.4f}')
    print(f'Late avg close_bonus (last 20):  {sum(late)/len(late):.4f}')
    print(f'Value loss range: {min(value_loss):.2f} - {max(value_loss):.2f}')

ssh.close()
