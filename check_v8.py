import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# Check if training is still running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep auto_train | grep -v grep')
ps_out = stdout.read().decode()
print("=== Process check ===")
if 'v8' in ps_out:
    print("v8 is RUNNING")
elif ps_out.strip():
    print(f"Something running: {ps_out.strip()[:200]}")
else:
    print("NO training process found!")

# Check log size and tail
stdin, stdout, stderr = ssh.exec_command('wc -l /root/RoboDuet/grasp_armonly_v8.log 2>/dev/null; echo "---"; tail -30 /root/RoboDuet/grasp_armonly_v8.log 2>/dev/null')
log_out = stdout.read().decode()
print("\n=== Log tail ===")
print(log_out)

# Get diagnostics
stdin, stdout, stderr = ssh.exec_command('grep "DIAG" /root/RoboDuet/grasp_armonly_v8.log 2>/dev/null')
diag_out = stdout.read().decode()
if diag_out.strip():
    print("=== Diagnostics ===")
    print(diag_out)

# Parse full log for metrics
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
print(f"\nTotal iterations parsed: {n}")

if n > 0:
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

    if n > 10:
        print(f"\n--- Latest 10 iterations ---")
        for i in range(max(0, n-10), n):
            tr = total_rew[i] if i < len(total_rew) else 0
            print(f"  iter {iters[i]}: close_bonus={close_bonus[i]:.4f}  total_rew={tr:.4f}  value_loss={value_loss[i]:.1f}")

    print(f"\nPeak close_bonus: {max(close_bonus):.4f} at iter {iters[close_bonus.index(max(close_bonus))]}")
    nonzero = sum(1 for v in close_bonus if v > 0)
    print(f"Non-zero close_bonus: {nonzero}/{n} ({100*nonzero/n:.1f}%)")
    print(f"Value loss range: {min(value_loss):.2f} - {max(value_loss):.2f}")

    if n > 200:
        first_q = close_bonus[:n//4]
        last_q = close_bonus[3*n//4:]
        print(f"\nclose_bonus trend: first_quarter={sum(first_q)/len(first_q):.4f} -> last_quarter={sum(last_q)/len(last_q):.4f}")
        first_vl = value_loss[:n//4]
        last_vl = value_loss[3*n//4:]
        print(f"value_loss trend:  first_quarter={sum(first_vl)/len(first_vl):.2f} -> last_quarter={sum(last_vl)/len(last_vl):.2f}")

ssh.close()
