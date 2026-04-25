# Defence Q&A Preparation
**AIMS 5790 Final Defence — Yuxiao Zhao**
*Read this the night before. Each answer is what you say out loud, not notes.*

---

## Quick-Reference Numbers (memorise these)

| Item | Value |
|---|---|
| Go1 mass / height | 12 kg / 30 cm |
| B2 mass / height | 60 kg / 55 cm |
| Z1 DOF | 6 + gripper |
| Total URDF links / DOF | 47 links, 19 DOF |
| Go1 Kp | 35 |
| B2 Kp / Kd | 200 / 20 |
| Parallel envs | 4096 |
| GPU | RTX 4090, 24 GB VRAM |
| Stage 1 | 1500 iter, ~3 h (arm only) |
| Stage 2 | 2000 iter, ~6 h (combined) |
| Fixed-base EE error | 4 cm |
| Standing + arm EE error | 6 cm |
| Target cube size | 6 cm |
| Gripper tip offset | 0.086 m along local X |
| Control frequency | 50 Hz |
| PPO clip | 0.2, LR 1e-3, γ=0.99, λ=0.95 |
| Real B2 cost | ~$30,000 USD |

---

## Category 1 — 创新点 / Contribution

**Q: What is your actual contribution? Did you just run someone else's code on a different robot?**

> "I want to be precise about what 'platform migration' means at a 5× mass scale.
> The B2 is not a drop-in replacement for the Go1.
> Every parameter in the system breaks: the joint PD gains had to go from Kp=35 to Kp=200 — nearly six times stiffer — because at the Go1 gain the B2 collapses in under one second.
> The standing pose had to be recomputed from scratch since Go1 angles cause the B2's longer legs to fold inward.
> The reward weights caused NaN gradients on B2 because the larger body creates much bigger initial EE errors.
> And I found and fixed a non-trivial simulator bug — IsaacGym silently merges zero-mass links, so ee_gripper_link disappears from the rigid body tensor; I had to reverse-engineer this and apply an 8.6 cm manual offset.
> None of this is documented.
> So the contribution is: a working B2+Z1 RoboDuet pipeline that did not exist before, with all the engineering decisions that make it work documented and reproducible."

**Q: Beyond engineering, is there any scientific novelty?**

> "This project is primarily an engineering contribution within a course context.
> The scientific insight is the empirical validation that RoboDuet's dual-policy architecture scales to a much heavier platform — the communication mechanism between locomotion and manipulation policies that worked at 12 kg also stabilises a 60 kg system.
> The fact that base height variance stays within ±1.4 cm with a 4 kg arm swinging on a 60 kg body is quantitative evidence that the guidance signal is doing real work.
> If I were writing a paper, I would add an ablation: remove the guidance channel and show performance degrades — that would be the scientific contribution."

---

## Category 2 — Baseline / Comparison

**Q: Why don't you compare against a monolithic policy? Without a baseline the results mean nothing.**

> "That is a fair point.
> The original RoboDuet paper by Pan et al. 2024 includes this ablation on the Go1: they show monolithic policy performs 50% worse on EE tracking accuracy.
> I explicitly chose to build on RoboDuet rather than implement a monolithic baseline myself because:
> first, replicating that ablation on the B2 would require implementing a second full training pipeline — roughly doubling the compute and time budget for a course project;
> second, my research question is specifically about platform migration, not about re-validating the dual-policy design choice.
> I acknowledge this limits the strength of my claims — I cannot say the dual-policy design is *why* B2 works, only that the whole system works."

**Q: How does your B2 result compare to the Go1 original results?**

> "Direct comparison is difficult because the tasks are not identical — the Go1 RoboDuet paper reports walking plus reaching, while my stable result is standing plus reaching.
> Within standing: Go1 achieves roughly 3–4 cm EE error; I achieve 4–6 cm.
> That slight degradation is expected: the B2's larger mass means any body perturbation has more inertia, adding noise to arm targeting.
> The fact that the degradation is only 2 cm rather than an order of magnitude suggests the architecture transfers well."

---

## Category 3 — Technical Definitions / Ambiguity

**Q: What exactly is the "guidance signal"? Can you be more specific?**

> "In the RoboDuet framework, at every 50 Hz control step, the arm policy outputs two things:
> one, the joint angle commands for the 6 arm joints;
> two, a latent vector — a compact encoding of the arm's current state and its planned near-future motion.
> This latent vector is fed as an additional input to the locomotion policy.
> Concretely, it tells the legs: the arm is about to swing right and forward, so pre-emptively shift your weight left.
> Without this vector, the arm's motion is a surprise to the legs every timestep, which causes reactive rather than anticipatory compensation — that is the source of instability in a naive two-controller design."

**Q: How do you define 'stable standing'? What is your metric?**

> "I define stable standing as: base height remains within ±1.4 cm of the target 0.55 m, and the robot does not fall — meaning no contact between the body links and the ground — over a 20-second evaluation rollout.
> The 20-second clip in the video is a hand-selected stable rollout, not an average.
> I am transparent about this: I do not claim all rollouts are equally stable.
> A more rigorous metric would be success rate over 100 random seeds, which I have not measured."

**Q: What is end-effector error — position only, or does it include orientation?**

> "Position error only — the L2 distance in metres between the computed gripper tip position and the target position in the world frame.
> Orientation matching is included in the reward function during training as a separate term, but I only report position error as the headline metric because the task is reaching a location, not achieving a precise grasp orientation.
> For a future grasping policy, orientation error would need to be reported separately."

**Q: You say 'fixed-base scripted grasp'. What does scripted mean — is the arm not using the learned policy?**

> "Good clarification. The grasp demo has two parts.
> The approach — moving the arm toward the cube — uses the learned PPO policy.
> The close-and-lift sequence is scripted: I command the gripper to close once the tip is within threshold distance, then command the arm to raise.
> The reason is that the current policy was trained only with an EE proximity reward; it learns to reach but not to close and lift.
> A fully learned grasp policy would require contact force feedback in the reward, which is future work."

---

## Category 4 — Training & Data

**Q: Why PPO and not SAC or TD3?**

> "Three reasons.
> First, PPO is on-policy, which is safer for high-dimensional robotics — it never replays stale data from a body configuration the robot no longer visits.
> Second, PPO's clipping mechanism — capping the probability ratio at 1 ± ε — directly prevents catastrophic policy updates, which is critical when a single bad update can destabilise a 60 kg robot's standing.
> Third, PPO is the standard in the RoboDuet codebase I built on, and changing the optimizer would have made attribution and debugging much harder for a course project.
> SAC can achieve better sample efficiency, but for massively parallel simulation where sample efficiency is less critical, PPO's stability advantages dominate."

**Q: How did you tune Kp=200, Kd=20? Is this principled or trial and error?**

> "Both. The starting point is physics: torque = Kp × position_error − Kd × velocity. Heavier robot, bigger errors, needs more torque, so Kp scales up. A rough back-of-envelope: B2 is 5× heavier than Go1, so Kp should be roughly 5–6× larger, giving ~175–210. I swept four values: 35, 100, 150, 200. At 35 the robot collapses in under 1 second. At 100 it wobbles and destabilises when the arm moves. At 150 it is stable but drifts over 20 seconds. At 200 with Kd=20 it holds the target height throughout. So the value is empirically validated within the range predicted by scaling analysis — it is not pure guesswork."

**Q: What is the observation space of each policy?**

> "The locomotion policy observes: body linear and angular velocity, base orientation quaternion, joint positions and velocities for all 12 leg joints, the velocity command from the user, and the guidance latent vector from the arm policy.
> The arm policy observes: body state — position and orientation — joint positions and velocities for all 6 arm joints, the current gripper tip position, and the target position in spherical coordinates — length, pitch, yaw.
> Both policies output target joint angles; these are converted to torques via PD control at 50 Hz."

**Q: Why only 2000 iterations for Stage 2? The manipulation reward was still climbing — did you undertrain?**

> "Yes, almost certainly. The learning curves show manipulation reward still rising at iteration 2000 with no sign of plateau. The limiting factor was wall-clock time — 6 hours on the remote RTX 4090 per run, and I needed multiple debugging iterations. Extending to 3000–4000 iterations is the single cheapest improvement available. Based on the trend, I estimate another 500–1000 iterations would reduce EE error from 6 cm to roughly 4–5 cm under standing conditions."

---

## Category 5 — Architecture Changes from Original RoboDuet

**Q: What did you change in the RoboDuet architecture versus the original paper?**

> "The network architecture and communication protocol are identical to the original. What I changed falls into three categories.
> First, environment configuration: Kp/Kd, default joint angles, simulation timestep parameters for the heavier body.
> Second, reward function weights: the manipulation weight had to be annealed during early training to prevent gradient explosion on B2's larger reach space; the original Go1 weights caused NaN gradients immediately.
> Third, observation preprocessing: specifically the gripper tip computation — I added the 0.086 m body-frame offset to compensate for the zero-mass link merging, which is a B2+Z1-specific fix not needed on the Go1.
> The dual-policy structure, the guidance signal dimension, and the PPO hyperparameters are unchanged."

**Q: The original RoboDuet paper shows walking plus manipulation. Your stable demo is only standing. Why?**

> "Stage 2 training does include walking commands, but in practice the policy learns to prioritise standing stability because the standing reward is easiest to exploit. When I run walking rollouts, the arm motion during a gait cycle creates larger CoM perturbations than the policy has learned to compensate — it degrades to short shuffles rather than sustained walking. This is a known training challenge: walking plus reaching requires much longer Stage 2 training and likely a more carefully shaped reward curriculum. It is the most important next step after fixing the lift command issue."

---

## Category 6 — Application / Limitations / Future

**Q: What are the real-world applications you envision for this platform?**

> "The B2+Z1 combination targets scenarios where a robot needs to both navigate and physically interact with objects in unstructured environments. Three concrete cases: disaster response — navigating rubble and manipulating debris or opening valves; warehouse automation in non-standardised environments where fixed-arm robots cannot reach; and elderly care assistance — carrying and handing objects in a home setting. The B2's 60 kg body and high payload capacity makes it more suitable than smaller platforms for tasks requiring physical strength, which is why the larger platform is worth the added control complexity."

**Q: Why no sim-to-real? The whole project is in simulation — how do you know any of this would work on real hardware?**

> "I cannot claim it would work without testing. Let me explain what sim-to-real would actually require. First, domain randomization: randomising mass ±20%, friction coefficients, motor delays, observation noise during training so the policy is not overfit to the exact simulated dynamics. I have not implemented this. Second, a deployment pipeline: getting the trained weights onto the B2's onboard computer, matching the control interface to the simulator's 50 Hz PD loop. Third and most importantly, safety: the real B2 costs approximately $30,000, and deploying an untested policy on a 60 kg robot without proper harness protocols and emergency stop infrastructure is genuinely dangerous. These are the reasons sim-to-real was out of scope for a single-semester course project — not lack of interest."

**Q: The pitch command doesn't produce vertical motion — this seems like a fundamental design flaw. Why didn't you fix it?**

> "It is a coordinate frame mismatch, not a fundamental flaw. The spherical coordinates — length, pitch, yaw — are defined in the robot's body frame, so pitch rotates around the robot's local horizontal axis. When the arm is extended forward, that axis is roughly parallel to world-Y, so increasing pitch produces lateral swing rather than world-Z lift. The fix is straightforward but requires time: either remap the target to world-frame Cartesian coordinates before feeding to the policy, or train a separate lift reward in world-Z. I identified the root cause, which is the engineering contribution; the fix is the next concrete task."

**Q: You selected a 'stable clip' for the standing video. Isn't that cherry-picking?**

> "It is selective presentation, and I disclosed this explicitly in the talk. The alternative — showing a random rollout that might fall — would be misleading in the other direction. The honest framing is: this policy can produce stable standing in simulation, and here is evidence of one such rollout. What I have not measured — and should for a paper — is the success rate over, say, 100 random seeds. That number would tell us whether we are looking at a reliable policy or a lucky rollout. I do not have that number, which is why I was careful not to claim robustness."

---

## Wildcard / Hard Questions

**Q: RoboDuet is a 2024 RAL paper. Isn't this too recent to base a course project on? How confident are you in the method?**

> "RAL is a peer-reviewed IEEE journal — the method has been reviewed and validated. The open-source code release with the paper means the implementation is the authors' own, reducing the risk of reimplementation bugs. My contribution is not to validate RoboDuet — that is done — but to test whether its design choices survive a large platform change. Using a recent, open-sourced, peer-reviewed method is the appropriate starting point for an applied course project."

**Q: How long would it realistically take to get a real-robot demo?**

> "Conservatively, 2–3 months assuming access to the B2 and a safety-equipped lab. Breakdown: 2–3 weeks for domain randomization retraining; 2 weeks for deployment pipeline setup and communication interface; 2–3 weeks for incremental testing starting with powered-on static tests, then harness-supported standing, then free standing, then arm motion. This is the roadmap I would bring to Professor Chen in the next discussion."

**Q: What would you do differently if you started over?**

> "Three things. First, implement the gripper tip offset fix and the standing pose computation at the very start, before any training — I lost significant time debugging these. Second, log success rate alongside reward curves from day one — I only have reward curves, not episode success rates, which limits how much I can say about reliability. Third, implement domain randomization in Stage 1, not as an afterthought — if it had been there from the beginning, the sim-to-real gap would be smaller today."

---

*Last updated: April 2026. If a professor asks something not covered here, the safe answer is: "That's a great question. The short answer is X. The honest longer answer is that I haven't measured/implemented that yet, and here is what it would take."*
