# Real Grasp Redesign Notes

- Current `b2z1_grasp_stand` is not object grasping training.
- Reward in `server_audit/rewards.py` shows arm policy tracks commanded EE pose/orientation.
- `use_vision=False`, so the policy observes `commands_arm_obs`, not a real box.
- Gripper is not in the arm action space, so the policy cannot learn grasp closure.

## Phase Plan

1. Real object reach/grasp env with stand-still dog
2. Fixed or scripted gripper close for phase 1
3. Add real object pose in EE frame to arm observation
4. Add grasp-specific rewards:
   - EE-to-object distance
   - gripper alignment
   - contact / enclosure proxy
   - object lift height
   - object drop penalty
5. After stable reach-to-object emerges, expand to learned gripper action
6. Then train quadruped balance with the new arm policy
7. Finally move to hybrid walking + grasping
