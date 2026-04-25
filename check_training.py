import paramiko
import time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

time.sleep(600)  # Wait 10 min

cmd = 'grep -E "close_bonus|Value function loss|Learning iteration|rew_total|rew_grasp_obj_dist|rew_grasp_xy_align|rew_grasp_z_align" /root/RoboDuet/grasp_armonly_v5.log'
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
print(f"Total iterations: {n}")

# Rolling averages
print(f"\n--- Rolling Average (window=100) ---")
window = 100
print(f"{'Window':>12} | {'close_bonus':>12} | {'total_rew':>10} | {'obj_dist':>10} | {'xy_align':>10} | {'z_align':>10} | {'val_loss':>10}")
print('-' * 95)
for start in range(0, n, 200):
    end = min(start + window, n)
    if end - start < 20:
        continue
    sz = end - start
    cb = sum(close_bonus[start:end]) / sz
    tr = sum(total_rew[start:end]) / sz if start < len(total_rew) else 0
    od = sum(obj_dist[start:end]) / sz if start < len(obj_dist) else 0
    xy = sum(xy_align[start:end]) / sz if start < len(xy_align) else 0
    za = sum(z_align[start:end]) / sz if start < len(z_align) else 0
    vl = sum(value_loss[start:end]) / sz
    print(f"  {start:>4}-{end-1:>4} | {cb:>12.4f} | {tr:>10.4f} | {od:>10.4f} | {xy:>10.4f} | {za:>10.4f} | {vl:>10.1f}")

# Latest sample
if n > 5:
    print(f"\n--- Latest 5 iterations ---")
    for i in range(n-5, n):
        tr = total_rew[i] if i < len(total_rew) else 0
        print(f"  iter {iters[i]}: close_bonus={close_bonus[i]:.4f}  total_rew={tr:.4f}  value_loss={value_loss[i]:.1f}")

# Overall stats
if n > 0:
    print(f"\nPeak close_bonus: {max(close_bonus):.4f} at iter {iters[close_bonus.index(max(close_bonus))]}")
    nonzero = sum(1 for v in close_bonus if v > 0)
    print(f"Non-zero close_bonus: {nonzero}/{n} ({100*nonzero/n:.1f}%)")

ssh.close()
