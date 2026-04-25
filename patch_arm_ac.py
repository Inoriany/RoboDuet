import paramiko
import time
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('jq1.9gpu.com', port=11360, username='root', password='QBCoP-ep', timeout=15)

# 1) Kill current training
print("Killing current training...")
stdin, stdout, stderr = ssh.exec_command('pkill -f auto_train_grasp_armonly')
time.sleep(5)

# 2) Modify arm_ac.py on remote to bound value output
ppo_dir = '/root/RoboDuet/go1_gym_learn/ppo_cse_automatic'
ac_path = f'{ppo_dir}/arm_ac.py'
stdin, stdout, stderr = ssh.exec_command(f'cat {ac_path}')
ac_content = stdout.read().decode()

# Find the evaluate method and add tanh bounding
old_eval = """    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.critic_history_encoder(obs_h)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        return value"""

new_eval = """    def evaluate(self, observation_history, privileged_observations, **kwargs):
        obs = observation_history[..., -self.num_obs:]
        obs_h = observation_history[..., :-self.num_obs]
        h_latent = self.critic_history_encoder(obs_h)
        value = self.critic_body(torch.cat((obs, privileged_observations, h_latent), dim=-1))
        # Bound value output to prevent bootstrap feedback explosion
        value = torch.tanh(value) * 30.0
        return value"""

if old_eval in ac_content:
    ac_content_new = ac_content.replace(old_eval, new_eval)
    sftp = ssh.open_sftp()
    with sftp.file(ac_path, 'w') as f:
        f.write(ac_content_new)
    sftp.close()
    print("Patched arm_ac.py: added tanh * 30.0 bound to value output")
else:
    print("ERROR: Could not find evaluate method in arm_ac.py!")
    # Try to find it
    for i, line in enumerate(ac_content.split('\n')):
        if 'evaluate' in line or 'critic_body' in line:
            print(f"  Line {i}: {line}")

# 3) Verify patch
stdin, stdout, stderr = ssh.exec_command(f'grep -n "tanh" {ac_path}')
print(f"Verify: {stdout.read().decode().strip()}")

ssh.close()
print("Done!")
