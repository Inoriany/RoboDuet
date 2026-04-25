import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

print('Waiting 150 seconds to reach ~500 iterations...')
time.sleep(150)

# Check process
stdin, stdout, stderr = ssh.exec_command('ps aux | grep auto_train | grep -v grep')
ps = stdout.read().decode()
print('Process running:', bool(ps.strip()))

# Get diagnostics
stdin, stdout, stderr = ssh.exec_command('grep "DIAG" /root/RoboDuet/grasp_armonly_v8.log')
diag = stdout.read().decode()
print('\n=== DIAGNOSTICS ===')
print(diag)

# Get metrics
cmd = 'grep -E "close_bonus|Value function loss|Learning iteration|rew_total|rew_grasp_obj_dist|rew_grasp_xy_align|rew_grasp_z_align" /root/RoboDuet/grasp_armonly_v8.log'
stdin, stdout, stderr = ssh.exec_command(cmd)
output = stdout.read().decode()

lines = output.strip().split('\n')
iters, close_bonus, value_loss, total_rew = [], [], [], []
obj_dist, xy_align, z_align = [], [], []
for line in lines:
    if 'Learning iteration' in line:
        iters.append(int(line.strip().split('iteration')[1].split('/')[0].strip()))
    elif 'close_bonus' in line:
        close_bonus.append(float(line.split(':')[-1].strip()))
    elif 'Value function loss' in line:
        value_loss.append(float(line.split(':')[-1].strip()))
    elif 'rew_total' in line:
        total_rew.append(float(line.split(':')[-1].strip()))
    elif 'rew_grasp_obj_dist' in line:
        obj_dist.append(float(line.split(':')[-1].strip()))
    elif 'rew_grasp_xy_align' in line:
        xy_align.append(float(line.split(':')[-1].strip()))
    elif 'rew_grasp_z_align' in line:
        z_align.append(float(line.split(':')[-1].strip()))

n = min(len(iters), len(close_bonus), len(value_loss))
print(f'\n=== SUMMARY ({n} iters, latest iter={iters[-1] if iters else "?"}) ===')

window = 50
print(f'\n{"Window":>12} | {"close_bonus":>12} | {"obj_dist":>10} | {"xy_align":>10} | {"z_align":>10} | {"total_rew":>10} | {"val_loss":>10}')
print('-' * 95)
for start in range(0, n, 100):
    end = min(start + window, n)
    if end - start < 10:
        continue
    sz = end - start
    cb = sum(close_bonus[start:end]) / sz
    od = sum(obj_dist[start:end]) / sz if start < len(obj_dist) else 0
    xy = sum(xy_align[start:end]) / sz if start < len(xy_align) else 0
    za = sum(z_align[start:end]) / sz if start < len(z_align) else 0
    tr = sum(total_rew[start:end]) / sz if start < len(total_rew) else 0
    vl = sum(value_loss[start:end]) / sz
    print(f'  {start:>4}-{end-1:>4} | {cb:>12.4f} | {od:>10.4f} | {xy:>10.4f} | {za:>10.4f} | {tr:>10.2f} | {vl:>10.2f}')

print(f'\nLatest 10:')
for i in range(max(0, n - 10), n):
    tr = total_rew[i] if i < len(total_rew) else 0
    print(f'  iter {iters[i]:4d}: close_bonus={close_bonus[i]:.4f}  total={tr:.2f}  val_loss={value_loss[i]:.2f}')

print(f'\nPeak close_bonus: {max(close_bonus):.4f} at iter {iters[close_bonus.index(max(close_bonus))]}')
print(f'Value loss range: {min(value_loss):.2f} - {max(value_loss):.2f}')

# Critical check: is close_bonus still improving or degrading?
q1 = close_bonus[:n//4]
q4 = close_bonus[3*n//4:]
trend = 'IMPROVING' if sum(q4)/len(q4) > sum(q1)/len(q1) else 'DEGRADING'
print(f'\nclose_bonus TREND: {trend}')
print(f'  Q1 avg: {sum(q1)/len(q1):.4f}')
print(f'  Q4 avg: {sum(q4)/len(q4):.4f}')

# Check dist from diagnostics - is arm actually getting closer?
print('\nDist from diagnostics:')
for d_line in diag.strip().split('\n'):
    if 'dist=' in d_line:
        parts = d_line.split('dist=')
        if len(parts) > 1:
            dist_val = parts[1].split()[0]
            iter_part = d_line.split('iter')[1].split(']')[0].strip()
            print(f'  iter {iter_part}: mean_dist={dist_val}')

ssh.close()
