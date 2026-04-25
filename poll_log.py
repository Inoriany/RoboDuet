"""Poll run_v11.log for completion."""
import paramiko

HOST = "jq1.9gpu.com"
PORT = 11360
USER = "root"
PASS = "QBCoP-ep"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

# Check if process still running
stdin, stdout, stderr = ssh.exec_command("ps aux | grep gen_grasp_v11 | grep -v grep")
ps_out = stdout.read().decode().strip()
if ps_out:
    print("STILL RUNNING:")
    print(ps_out)
else:
    print("Process finished.")

# Read log
stdin, stdout, stderr = ssh.exec_command("cat /root/RoboDuet/run_v11.log")
log = stdout.read().decode()
print("\n=== LOG ===")
print(log[-3000:] if len(log) > 3000 else log)

ssh.close()
