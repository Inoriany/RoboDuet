import os

import paramiko
from imageio import v2 as iio


HOST = "jq1.9gpu.com"
PORT = 11360
USER = "root"
PASS = "QBCoP-ep"

REMOTE_VIDEO = "/root/RoboDuet/b2z1_grasp_fixedbase_success.mp4"
LOCAL_VIDEO = r"D:\CUHK\AIMS_5790\b2z1_grasp_fixedbase_v2.mp4"
OUT_DIR = r"D:\CUHK\AIMS_5790\video_contact_sheet\fixedbase_v2"


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)

    stdin, stdout, stderr = ssh.exec_command(
        "while pgrep -f gen_grasp_fixedbase_success.py >/dev/null; do sleep 2; done; ls -l /root/RoboDuet/b2z1_grasp_fixedbase_success.mp4"
    )
    print(stdout.read().decode(errors="ignore"))
    err = stderr.read().decode(errors="ignore")
    if err.strip():
        print(err)

    sftp = ssh.open_sftp()
    sftp.get(REMOTE_VIDEO, LOCAL_VIDEO)
    sftp.close()
    ssh.close()

    print("DOWNLOADED", os.path.getsize(LOCAL_VIDEO))
    os.makedirs(OUT_DIR, exist_ok=True)
    rdr = iio.get_reader(LOCAL_VIDEO)
    for idx in [0, 100, 220, 300, 420, 570]:
        frame = rdr.get_data(idx)
        path = os.path.join(OUT_DIR, f"frame_{idx:03d}.png")
        iio.imwrite(path, frame)
        print(path)
    rdr.close()


if __name__ == "__main__":
    main()
