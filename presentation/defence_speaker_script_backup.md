# Defence Speaker Script Backup

Standalone backup copy of the PPT speaker notes.

## Slide 1: Slide 1

Good morning, professors. My name is Yuxiao Zhao, student ID 1155246057. Today I present my AIMS 5790 final project — adapting a legged manipulation framework called RoboDuet from a small 12-kilogram robot to a much larger 60-kilogram platform called the B2. My supervisor is Professor Fei Chen. The presentation is about 25 minutes, and I am happy to take questions afterwards.

## Slide 2: Slide 2

Here is the structure. I will start with background on what a legged manipulator is and why it is challenging. Then I will show the full project pipeline — every step from building the robot model to the final demo. After that I go into hardware and software setup, explain the RoboDuet method, walk through training and results, show simulation videos, and finally discuss what has been achieved and what remains as future work. About 25 minutes total for the presentation, with time reserved for questions.

## Slide 3: Slide 3

Let me start with what we are building. In robotics, there are two fundamental capabilities. Locomotion — walking, climbing stairs, navigating rough terrain. And manipulation — picking things up, pushing buttons, opening doors. Most robots do one well but not both. Industrial robot arms can manipulate very precisely, but they are bolted to the floor and cannot move. Quadruped robots like the Unitree family can walk and even run, but they have no arms and cannot interact with objects. A loco-manipulator combines both: a walking robot with a robotic arm mounted on its back. The analogy I like: think of a person carrying a tray of drinks through a crowded room. Your legs handle the walking, your arms handle the tray, and your brain coordinates them so you do not trip or spill. That coordination problem is exactly what we are solving — except we use artificial intelligence instead of a human brain.

## Slide 4: Slide 4

Why not just control legs and arm separately? Three reasons. First, when the arm moves it shifts the centre of gravity and can destabilise the body. Second, Newton's third law means the arm creates reaction forces that the legs must compensate in real time. Third, walking vibrations shake the arm and ruin its precision. A single giant controller for all joints has too many parameters and does not converge. Two independent controllers ignore each other and the system is unstable. We need something in between — two controllers that actively communicate. That is the core idea of RoboDuet, which I will explain in detail in Section 4.

## Slide 5: Slide 5

We train the robot using reinforcement learning. In RL, the robot — our agent — takes actions in a physics simulator, receives a numerical reward after each action, and over millions of trials learns a policy — a mapping from sensor readings to joint actions — that maximises total reward. The algorithm is PPO — Proximal Policy Optimisation — introduced by Schulman et al. in 2017 and now the standard for robotics RL. PPO's key idea is the clipping mechanism: when we update the policy, if the update is too aggressive — the probability ratio deviates too far from 1.0 — it gets clipped. This prevents catastrophic policy changes and keeps training stable. Critically, we do all of this in simulation. NVIDIA IsaacGym runs 4096 robot copies simultaneously on a single GPU. Three hours of wall-clock time equals about 1.5 years of real-robot experience — that massive parallelism is what makes RL practical for robotics.

## Slide 6: Slide 6

My goal: take RoboDuet, validated on the 12-kilogram Go1, and make it work on the 60-kilogram B2 — a 5-times mass increase, 25 centimetres taller. Every parameter changes: joint limits, control gains, reward weights, standing pose. The Go1 pre-trained weights are useless on the B2, so I had to retune everything from scratch. The arm is the same Unitree Z1 with 6 degrees of freedom plus a gripper, but the dynamics change dramatically on a body five times heavier.

## Slide 7: Slide 7

This flowchart shows the complete pipeline in five phases. Phase 1: merge the B2 body and Z1 arm into a single URDF file — 47 links, 19 degrees of freedom. Phase 2: configure IsaacGym with Kp 200, Kd 20, and 4096 parallel environments on the GPU. Phase 3: Stage 1 training — the arm learns to reach while the body is fixed. 1500 iterations, about 3 hours. Phase 4: Stage 2 training — legs and arm train together with communication enabled. 2000 iterations, about 6 hours. Phase 5: generate demo videos and measure accuracy — 4 to 6 centimetres end-effector error. I will now walk through each phase in detail.

## Slide 8: Slide 8

The Go1 weighs 12 kilograms and stands 30 centimetres tall. The B2 weighs 60 kilograms and stands 55 centimetres. Both have 12 leg degrees of freedom and the same Z1 arm. The critical difference is control gains. The Go1 uses Kp equals 35 — very soft gains for a lightweight robot. For the B2, I had to increase Kp to 200 and Kd to 20 — nearly six times stiffer. Without these higher gains, the B2 cannot support its own weight and collapses immediately. On the right you can see the B2 with the Z1 arm mounted on top. Same arm, completely different body — every mechanical parameter had to change.

## Slide 9: Slide 9

The core simulator is NVIDIA IsaacGym — a GPU-accelerated physics engine simulating rigid body dynamics entirely on the GPU, with no CPU transfer bottleneck. It runs 4096 robot instances simultaneously. The URDF file is the robot blueprint: links, joints, masses, meshes in XML format. Our b2z1.urdf was created by manually merging B2 and Z1 descriptions. Training ran on a remote RTX 4090 with 24 gigabytes of VRAM, PyTorch 2.4.1 with CUDA 12.1. I wrote automation scripts to upload code, launch training, and poll logs for completion. One undocumented gotcha: IsaacGym must be imported before PyTorch in the code, or the simulator crashes. This took considerable time to figure out.

## Slide 10: Slide 10

PD control converts target joint angles into torques: torque equals Kp times position error minus Kd times velocity. I swept four Kp values. At 35 — the Go1 default — the B2 collapses immediately. At 100, it wobbles badly and arm motion destabilises it. At 150, stable but drifts over 20 seconds. At 200 with Kd 20, the robot stands stably even with the arm moving aggressively. This scaling is physically intuitive: five times heavier needs roughly six times stiffer joints. I also had to recompute the default joint angles since Go1 angles cause the B2's longer legs to fold incorrectly.

## Slide 11: Slide 11

An important technical detail about computing the end-effector position. The simulator stores all rigid bodies in a flat tensor — one row per body. To find the gripper tip, I need the correct row index. The B2 trunk is index 0, the 12 leg links are indices 1 through 12, the Z1 arm links are 13 through 24, and gripperMover — the moving jaw — is index 25, which is the last physical rigid body. Now here is the catch: in the URDF file, there is a link called ee_gripper_link at link number 47. This should be the exact gripper tip position. But this link has zero mass. When IsaacGym processes the URDF, it silently merges any zero-mass links into their parent body. So ee_gripper_link does not exist in the rigid body tensor — reading its position gives garbage. The fix: I read the position and quaternion of body 25 and manually add a 0.086-metre offset along its local X axis. Without this 8.6-centimetre correction, the computed position is behind the actual tip. For a 6-centimetre cube target, that is the difference between reaching and missing completely.

## Slide 12: Slide 12

RoboDuet was published by Pan et al. in 2024 in IEEE Robotics and Automation Letters. Instead of one monolithic controller or two independent ones, it uses two policies that actively communicate. The locomotion policy controls 12 leg joints; the arm policy controls 6 arm joints plus the gripper. At every 50 Hz control step, the arm sends a guidance signal to the legs — essentially a preview of what it is about to do, so the legs can prepare in advance. On the original Go1, RoboDuet achieved 50 percent better tracking accuracy versus a monolithic policy, validated both in simulation and on real hardware. The open-source code made it suitable for my B2 migration.

## Slide 13: Slide 13

Training has two stages. Stage 1: policies train independently. Legs learn to walk with the arm frozen — just a static load. The arm learns to reach with the body fixed — no body motion to worry about. Each sub-task is simpler and converges quickly. Stage 2: both policies are unfrozen and trained together with communication channels activated. The arm sends guidance signals to the legs, and the legs send body state back. The reward covers both locomotion and manipulation. This curriculum is crucial — without it, the arm's random movements destabilise the legs before either policy learns anything, and training diverges.

## Slide 14: Slide 14

Let me show you Stage 1 in practice. [CLICK TO PLAY VIDEO] The body is fixed to the ground. The arm receives targets in spherical coordinates — length, pitch, yaw — and smoothly transitions between reaching poses. It extends forward, moves sideways, pulls back, all controlled by the neural network. This is the foundation: the arm must first learn accurate reaching before combining with leg motion. This checkpoint serves as the starting point for both Stage 2 cooperative training and the grasp approaching demo I show later.

## Slide 15: Slide 15

This table shows the signal flow between the two policies. The human gives velocity commands to the legs and spherical coordinates to the arm. The key innovation is inter-agent communication: the arm computes a guidance signal telling the legs what it is about to do — for example, I am about to swing right, so shift weight left. The legs send body state back — the body is tilted 3 degrees, so adjust your aim. Think of a waiter bracing their stance before lifting a heavy plate. Without these guidance signals, the arm surprises the legs every time and the system becomes unstable.

## Slide 16: Slide 16

The reward function has three groups. Locomotion: positive reward for velocity tracking, target height, and level orientation. Manipulation: reward for end-effector proximity — weight 3.0, the highest — and orientation matching. Regularisation: penalties for excessive torque, jerky motions, joint limits, and self-collision. When I applied Go1 reward weights to the B2, training produced NaN gradients. The B2's larger reach range created enormous initial errors, and weight 3.0 produced gradient explosions. The fix was to reduce manipulation weight during early iterations and gradually restore it.

## Slide 17: Slide 17

Stage 1 trains the arm alone for 1500 iterations, about 3 hours on RTX 4090, producing an arm checkpoint. Stage 2 takes that checkpoint and trains both policies together for 2000 iterations, about 6 hours, with communication enabled. Hyperparameters: Adam optimiser, learning rate 10 to the minus 3, PPO clip 0.2, gamma 0.99, GAE lambda 0.95. We run 4096 environments each collecting 24 steps, reused for 5 epochs with 4 mini-batches, fitting the 24 gigabyte VRAM budget.

## Slide 18: Slide 18

Left: total return increases monotonically — PPO training is stable with no collapses. This was not guaranteed since we are on a completely different platform from the Go1 that these reward weights were designed for. Right: locomotion reward saturates after about 500 iterations — walking is simpler. The manipulation reward keeps climbing through all 2000 iterations without plateauing, suggesting longer training would further improve reaching accuracy. Extending to 3000 or 4000 iterations would be a straightforward way to improve results given more compute time.

## Slide 19: Slide 19

Here are the quantitative results. In the fixed-base arm reaching demo, the base height is exactly 0.55 metres — zero variation because the body is bolted down. The end-effector position error is 4 centimetres. In the standing plus arm reaching demo, the base height averages 0.52 metres — slightly lower because the arm weight pulls the body down. The height variation is only plus or minus 1.4 centimetres over the full 20-second demo. For a 60-kilogram robot with a 4-kilogram arm swinging on its back, that is remarkably stable. End-effector error increases to 6 centimetres with a free base, since body motion adds noise to targeting. Both values — 4 and 6 centimetres — are within our 6-centimetre cube size, confirming the arm reliably reaches the target vicinity.

## Slide 20: Slide 20

I want to be transparent about the engineering problems I encountered, because they show the practical depth of this work. Problem one: IsaacGym crashes if PyTorch is imported first — there is an undocumented requirement that the isaacgym module must initialise the GPU context before PyTorch claims it. Problem two: the robot collapsed immediately on spawn because the default joint angles were designed for the Go1's short legs — applied to the B2's longer legs, the knees folded inward. I had to compute new neutral angles for the B2's 0.55-metre height. Problem three: increasing the pitch command makes the arm swing sideways instead of up — I will explain why shortly. Problem four: end-effector position readings were garbage because the zero-mass ee_gripper_link gets silently merged out by IsaacGym. Problem five: NaN gradients from Go1 reward weights being too aggressive for the B2's larger reach range. Each of these took significant debugging time.

## Slide 21: Slide 21

[CLICK TO PLAY VIDEO] This standing demo uses a deliberately selected stable clip. The main point is visual: throughout the shown segment, the robot remains upright while the arm changes pose. I am using this as evidence that a stable standing example exists in simulation, while avoiding any claim that all standing rollouts are equally reliable.

## Slide 22: Slide 22

Four frames from the updated standing clip at different time points: 0, 4, 8, and 12 seconds. Across all four frames, the body remains upright while the arm changes pose. This slide is only meant to mirror the selected stable video segment on the previous slide.

## Slide 23: Slide 23

[CLICK TO PLAY VIDEO] This is the most important demo. The arm approaches a 6-centimetre green cube placed at the calibrated reach target. The base is fixed for stable camera framing. As the video plays, the gripper moves toward the cube, closes, and then the object follows the gripper upward. This demonstrates that the fixed-base scripted grasp sequence now works visually in simulation. However, I still want to be precise about the project status: this is a demo pipeline result, not yet a fully learned robust grasp policy under floating-base locomotion. The remaining challenge is to make grasping and lifting reliable under the full RoboDuet standing or walking setting.

## Slide 24: Slide 24

Key frames from the approaching demo. Frame 0: arm retracted, cube on the ground. Around frame 320, the gripper is closing around the cube. Around frame 560, the object has been lifted upward with the gripper. So the fixed-base visual demo now shows the complete sequence of approach, close, and lift. The important research message is that accurate reaching on B2 was achieved through reinforcement learning, and this created the foundation needed for a grasp-and-lift style demonstration in the fixed-base setting.

## Slide 25: Slide 25

This slide explains why general learned lifting is still hard even though the fixed-base demo can show a lift. The arm commands use spherical coordinates — length, pitch, yaw — all defined relative to the robot's body frame. Length controls how far the arm extends. Yaw controls left-right rotation. Pitch should control the up-down angle. But when I increase pitch from 0.25 to 0.50, the arm does NOT go up — it swings sideways by about 35 centimetres in the Y direction. Why? Because pitch rotates about the body's local axis, which is roughly horizontal when the arm is extended forward. So increasing pitch rotates the arm around a horizontal axis, producing lateral motion, not true vertical control. For a robust learned lift policy, we need either world-frame commands, scripted joint trajectories for the lift phase, or a new lift-specific reward. That is the real technical blocker now.

## Slide 26: Slide 26

Five accomplishments. First, reproduced RoboDuet on the original Go1 to establish a baseline. Second, migrated the complete pipeline to B2 — new URDF, PD gains, reward weights, standing pose, all from scratch. This was the bulk of the engineering work. Third, trained two checkpoints: arm-only reaching and standing with arm reaching, using the two-stage procedure. Fourth, demonstrated the arm reaching a 6-centimetre target with 4 to 6 centimetres error in simulation. Fifth, delivered a fully reproducible codebase with automated training and video-generation scripts.

## Slide 27: Slide 27

I want to be completely transparent about the remaining limitations. First, robust learned grasping is not yet solved. The fixed-base demo can show close and lift, but the current policy does not yet perform repeatable grasping under standing or walking disturbances. Second, general learned lifting is not solved — as I just explained, the spherical coordinate command system does not provide clean world-Z control. Third, walking plus manipulation is not yet stable — the Stage 2 policy works for standing with arm reaching, but walking while reaching introduces too much perturbation. Fourth, there is no sim-to-real transfer. The real B2 costs approximately 30 thousand dollars and was not available. Additionally, sim-to-real requires domain randomisation which I have not yet implemented, and a 60-kilogram robot with an untested policy is genuinely dangerous without proper safety protocols.

## Slide 28: Slide 28

Here is the roadmap for future work. First, fix grasping by synchronising the gripper tip position with object placement more precisely. Second, implement vertical lifting — either by remapping arm commands to world-frame Z coordinates, or by using scripted joint trajectories for the lift phase. Third, train a contact-based grasp policy that uses contact forces and friction as reward signals, with curriculum learning. Fourth, retrain Stage 2 with walking plus reaching plus grasping together. Fifth, implement domain randomisation for eventual sim-to-real transfer — randomise mass, friction, motor delays so the policy is robust to the reality gap. Sixth, replace fixed commands with vision-based target detection using an onboard camera.

## Slide 29: Slide 29

All deliverables: the written report compiled from LaTeX, this presentation with three embedded demo videos, the b2z1.urdf robot model, training and demo generation scripts that are fully reproducible, and training logs with complete tensorboard data.

## Slide 30: Slide 30

Thank you very much for your time. I am happy to answer any questions — about the RL training, the hardware migration, the software challenges, or future work directions. Thank you.
