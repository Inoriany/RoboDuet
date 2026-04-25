import paramiko
import os
import time


HOST = os.environ.get("ROBODUET_SSH_HOST")
PORT = int(os.environ.get("ROBODUET_SSH_PORT", "22"))
USER = os.environ.get("ROBODUET_SSH_USER", "root")
PASS = os.environ.get("ROBODUET_SSH_PASSWORD")


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=40)

    sftp = ssh.open_sftp()
    sftp.put(r"D:\CUHK\AIMS_5790\gen_grasp_fixedbase_success.py", "/root/RoboDuet/gen_grasp_fixedbase_success.py")
    sftp.close()

    for cmd in [
        "pkill -f gen_grasp_fixedbase_success.py || true",
        "rm -f /root/RoboDuet/run_fixedbase_success.log",
        "nohup bash /root/RoboDuet/run_fixedbase_success.sh &",
    ]:
        ssh.exec_command(cmd)
        time.sleep(1)

    time.sleep(4)
    stdin, stdout, stderr = ssh.exec_command("tail -n 120 /root/RoboDuet/run_fixedbase_success.log 2>/dev/null || true")
    print(stdout.read().decode(errors="ignore"))
    err = stderr.read().decode(errors="ignore")
    if err.strip():
        print(err)

    ssh.close()


if __name__ == "__main__":
    main()
