# RoboDuet on Unitree B2 + Z1

Platform migration and simulation evaluation of a RoboDuet-style legged manipulation framework from the lightweight Unitree Go1 platform to a much heavier Unitree B2 quadruped with a Unitree Z1 arm.

This repository contains the B2+Z1 robot description, training/evaluation scripts, presentation source, and engineering notes for an AIMS 5790 final project supervised by Prof. Fei Chen at The Chinese University of Hong Kong.

## Project Summary

Legged manipulation combines two difficult robotics capabilities: locomotion and object manipulation. A quadruped must keep balance while an arm mounted on its body reaches, pushes, or grasps. The arm shifts the robot's center of mass and creates reaction forces that the legs must compensate for in real time.

This project adapts the RoboDuet framework, originally demonstrated on the 12 kg Unitree Go1, to the 60 kg Unitree B2 platform with a Z1 robotic arm. The main challenge is not simply changing a URDF file: the heavier body requires new PD gains, standing pose, reward tuning, training scripts, and debugging of simulator-specific issues.

## Key Contributions

- Built a combined B2 + Z1 robot model for Isaac Gym simulation.
- Retuned joint PD control for the 60 kg B2 platform, increasing stiffness from the Go1 default `Kp=35` to `Kp=200, Kd=20`.
- Recomputed stable default standing posture for the B2 body.
- Trained fixed-base Z1 reaching and whole-body standing-with-arm-reaching policies using PPO.
- Debugged Isaac Gym rigid-body indexing for the gripper tip, including the zero-mass link merge issue and a `0.086 m` local-frame offset correction.
- Produced simulation demos for fixed-base reaching, stable standing, and fixed-base grasp-and-lift visualization.
- Documented current limitations honestly: no real-robot deployment yet, no domain randomization yet, and robust walking-plus-grasping remains future work.

## Results

| Evaluation setting | Result |
|---|---:|
| Fixed-base Z1 reaching error | ~4 cm |
| Standing + arm reaching error | ~6 cm |
| Standing demo duration | ~20 s stable selected rollout |
| Target cube size | 6 cm |
| Parallel Isaac Gym environments | 4096 |
| Stage 1 training | 1500 iterations, ~3 hours |
| Stage 2 training | 2000 iterations, ~6 hours |

These results are simulation-only. The fixed-base grasp demo is a visualization pipeline result, not a fully learned robust grasp policy under locomotion disturbances.

## Repository Structure

```text
.
├── b2z1.urdf / b2z1.xml              # Combined B2 + Z1 robot descriptions
├── meshes/                           # Robot mesh assets used by the URDF
├── auto_train_*.py                   # Training launch scripts and PPO experiments
├── restart_*.py                      # Resume/restart helpers for staged training
├── gen_grasp_*.py                    # Grasp and reaching demo generation scripts
├── gen_video*.py                     # Video/rendering utilities
├── real_grasp_env.py                 # Grasping environment prototype
├── real_grasp_rewards.py             # Reward definitions for grasping experiments
├── remote_snapshot/                  # Important code snapshots from remote training
├── server_audit/                     # Selected source/config snapshots from training runs
├── presentation/                     # LaTeX presentation source and PPT generation tools
├── realrobot_proposal/               # Initial real-robot testing discussion proposal
└── README.md
```

Generated videos, compiled PDFs, PowerPoint files, large training logs, Python caches, and local helper files are intentionally excluded from Git.

## Technical Background

### RoboDuet-Style Control

RoboDuet uses two communicating policies rather than one monolithic controller:

- A locomotion policy controls the 12 leg joints.
- A manipulation policy controls the Z1 arm and gripper.
- The arm policy sends a guidance signal to the locomotion policy so the legs can compensate for upcoming arm motion.

This communication is important because arm motion changes the center of mass and can destabilize a quadruped if the legs react too late.

### PD Control

The robot actions are target joint angles converted into torque by PD control:

```text
torque = Kp * (target_angle - current_angle) - Kd * joint_velocity
```

- `Kp` controls joint stiffness and corrective force.
- `Kd` provides damping and reduces oscillation.

The original Go1 gains were too soft for the B2. With `Kp=35`, the B2 collapses. The tuned setting `Kp=200, Kd=20` supports stable standing in simulation.

## Training Pipeline

The project follows a two-stage curriculum:

1. **Stage 1: Arm-only reaching**
   - The robot base is fixed.
   - The Z1 arm learns to reach target positions.
   - This isolates manipulation from locomotion instability.

2. **Stage 2: Whole-body standing + arm reaching**
   - Locomotion and arm policies are trained together.
   - Inter-policy communication is enabled.
   - The objective combines standing stability and end-effector tracking.

## Important Scripts

| File | Purpose |
|---|---|
| `auto_train_kp200.py` | B2 training entry point with tuned PD settings |
| `auto_train_grasp_armonly.py` | Arm-only reaching/grasp approach training |
| `auto_train_grasp_fixedbase.py` | Fixed-base grasp/reaching experiments |
| `restart_phase1_curriculum.py` | Curriculum restart helper |
| `gen_grasp_fixedbase_success.py` | Generate the fixed-base grasp demo |
| `gen_video_from_checkpoint.py` | Render a policy rollout from a checkpoint |
| `real_grasp_env.py` | Grasping environment prototype |
| `real_grasp_rewards.py` | Reward shaping terms for grasping |
| `presentation/gen_pptx.py` | Generate PPTX from LaTeX slides and embedded media |

## Environment

The project was developed around:

- Python 3.8+
- PyTorch 2.4.1
- CUDA 12.1
- NVIDIA Isaac Gym
- RTX 4090, 24 GB VRAM
- 4096 parallel simulation environments

Exact dependency setup depends on the Isaac Gym installation path and the RoboDuet base code. Isaac Gym must be imported before PyTorch in scripts that use both libraries; otherwise GPU context initialization may fail.

## Usage Notes

This repository is primarily a research artifact and project archive. Many scripts assume the RoboDuet/Isaac Gym runtime layout used during development. Before running training, check paths inside the script you intend to use.

Typical workflow:

```bash
# 1. Prepare Isaac Gym and RoboDuet dependencies
# 2. Verify that b2z1.urdf and meshes/ are discoverable by the simulator
# 3. Launch a training script
python auto_train_kp200.py

# 4. Render a checkpoint rollout
python gen_video_from_checkpoint.py
```

## Current Limitations

- No real B2+Z1 hardware deployment has been performed.
- No domain randomization has been implemented yet.
- Walking plus manipulation is not yet robust.
- The fixed-base grasp demo includes scripted close/lift logic and should not be interpreted as a fully learned grasping policy.
- The stable standing video is a selected stable rollout, not a measured success rate over many random seeds.

## Future Work

- Add domain randomization for mass, friction, latency, and observation noise.
- Measure success rate over many random seeds, not only selected rollouts.
- Replace body-frame spherical arm commands with world-frame Cartesian targets for more reliable vertical lifting.
- Train contact-aware grasping rewards using object contact and lift success.
- Deploy cautiously on real hardware with harness, emergency stop, and staged safety checks.
- Add vision-based target detection for more realistic manipulation tasks.

## Acknowledgements

This project builds conceptually on RoboDuet by Pan et al. and was completed as part of AIMS 5790 at The Chinese University of Hong Kong. Thanks to Prof. Fei Chen for supervision and guidance.

## Safety Notice

Policies trained only in simulation should not be deployed directly on a real 60 kg robot. Real-robot testing requires domain randomization, low-gain startup procedures, mechanical support or harnessing, emergency stop readiness, and supervision by experienced operators.
