#!/usr/bin/env python3
"""
B2Z1 Robot Simulation Video Generator
Visualizes B2Z1 quadruped-arm robot walking and manipulating
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
import os

# Robot parameters
BODY_WIDTH = 0.4
BODY_LENGTH = 0.8
LEG_LENGTH = 0.5
ARM_LENGTH_1 = 0.3
ARM_LENGTH_2 = 0.3
TIME_STEPS = 300  # 5 seconds at 60 fps

def get_leg_position(leg_index, time, gait_phase=0):
    """
    Calculate leg position for a quadruped gait
    leg_index: 0=FR, 1=FL, 2=BR, 3=BL
    """
    # Trotting gait: front-right and back-left move together, etc
    phase_offset = (leg_index // 2) * np.pi
    motion_phase = 2 * np.pi * (time / TIME_STEPS) + phase_offset + gait_phase
    
    # X: forward motion
    x_base = -BODY_LENGTH / 2 + (leg_index % 2) * BODY_LENGTH
    x_offset = 0.2 * np.sin(motion_phase)
    x = x_base + x_offset
    
    # Y: lateral position
    y_base = -BODY_WIDTH / 2 + (leg_index // 2) * BODY_WIDTH
    y = y_base
    
    # Z: leg height (bouncing)
    z_base = -LEG_LENGTH
    # Swing phase: leg lifts up, stance phase: leg down
    swing = (np.sin(motion_phase) > 0)  # True in swing phase
    z_offset = 0.15 * swing  # Leg lifts 0.15 units
    z = z_base + z_offset
    
    return np.array([x, y, z])

def get_arm_position(time, reach_target=True):
    """Calculate arm joint positions for grasping motion"""
    # Arm motion: extend, grasp, retract
    cycle = 2 * np.pi * (time / TIME_STEPS / 2)  # Half speed for arm
    
    # Shoulder (relative to body front)
    shoulder = np.array([0.3, 0, 0])
    
    # Upper arm angle (shoulder pitch)
    shoulder_pitch = np.pi/6 + 0.3 * np.sin(cycle)
    
    # Elbow angle
    elbow_pitch = np.pi/4 + 0.4 * np.sin(cycle)
    
    # Calculate end effector
    elbow_x = shoulder[0] + ARM_LENGTH_1 * np.cos(shoulder_pitch)
    elbow_z = shoulder[2] - ARM_LENGTH_1 * np.sin(shoulder_pitch)
    
    wrist_x = elbow_x + ARM_LENGTH_2 * np.cos(shoulder_pitch - elbow_pitch)
    wrist_z = elbow_z - ARM_LENGTH_2 * np.sin(shoulder_pitch - elbow_pitch)
    
    return shoulder, (elbow_x, 0, elbow_z), (wrist_x, 0, wrist_z)

def create_robot_visualization():
    """Create a 3D animation of the B2Z1 robot"""
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_xlim([-1, 1])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-1.2, 0.5])
    ax.set_xlabel('X (Forward)', fontsize=10)
    ax.set_ylabel('Y (Lateral)', fontsize=10)
    ax.set_zlabel('Z (Height)', fontsize=10)
    ax.set_title('B2Z1 Quadruped-Arm Robot Simulation', fontsize=14, fontweight='bold')
    
    # Add ground
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-0.8, 0.8, 10))
    zz = np.zeros_like(xx) - 1.0
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='brown')
    
    # Plot elements
    body_line, = ax.plot([], [], [], 'b-', linewidth=4, label='Body')
    legs_line = [ax.plot([], [], [], 'g-', linewidth=2)[0] for _ in range(4)]
    leg_feet = [ax.scatter([], [], [], c='green', s=50) for _ in range(4)]
    
    arm_shoulder_to_elbow, = ax.plot([], [], [], 'r-', linewidth=3, label='Arm (Upper)')
    arm_elbow_to_wrist, = ax.plot([], [], [], 'r--', linewidth=2, label='Arm (Lower)')
    wrist_point = ax.scatter([], [], [], c='red', s=100, marker='o')
    
    # Target object
    target_object = ax.scatter([], [], [], c='orange', s=200, marker='s', label='Target Object')
    
    # Text information
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11)
    status_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10)
    
    ax.legend(loc='upper right', fontsize=9)
    
    def update(frame):
        time = frame
        
        # Body position (center)
        body_center = np.array([0.05 * np.sin(2 * np.pi * time / TIME_STEPS), 0, 0])
        
        # Body bounds
        body_front = body_center + np.array([BODY_LENGTH/2, 0, 0])
        body_back = body_center - np.array([BODY_LENGTH/2, 0, 0])
        
        body_line.set_data([body_back[0], body_front[0]], [body_back[1], body_front[1]])
        body_line.set_3d_properties([body_back[2], body_front[2]])
        
        # Update legs
        leg_colors = ['FR', 'FL', 'BR', 'BL']
        for leg_idx in range(4):
            # Get leg foot position
            foot_pos = get_leg_position(leg_idx, time) + body_center
            
            # Hip position (attachment to body)
            hip_x = -BODY_LENGTH/4 + (leg_idx % 2) * BODY_LENGTH/2
            hip_y = -BODY_WIDTH/2 + (leg_idx // 2) * BODY_WIDTH
            hip_pos = body_center + np.array([hip_x, hip_y, 0])
            
            # Draw leg line
            legs_line[leg_idx].set_data([hip_pos[0], foot_pos[0]], [hip_pos[1], foot_pos[1]])
            legs_line[leg_idx].set_3d_properties([hip_pos[2], foot_pos[2]])
            
            # Draw foot
            leg_feet[leg_idx]._offsets3d = ([foot_pos[0]], [foot_pos[1]], [foot_pos[2]])
        
        # Update arm
        shoulder, elbow, wrist = get_arm_position(time)
        shoulder += body_center
        elbow = np.array(elbow) + body_center
        wrist = np.array(wrist) + body_center
        
        # Upper arm
        arm_shoulder_to_elbow.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        arm_shoulder_to_elbow.set_3d_properties([shoulder[2], elbow[2]])
        
        # Lower arm
        arm_elbow_to_wrist.set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
        arm_elbow_to_wrist.set_3d_properties([elbow[2], wrist[2]])
        
        # Wrist
        wrist_point._offsets3d = ([wrist[0]], [wrist[1]], [wrist[2]])
        
        # Target object position
        target_x = 0.6
        target_z = -0.8
        grasp_cycle = 2 * np.pi * (time / TIME_STEPS / 2)
        dist_to_wrist = np.linalg.norm(wrist - np.array([target_x, 0, target_z]))
        if dist_to_wrist < 0.15:  # Close to grasping
            grasp_phase = "GRASPING"
        elif time % (TIME_STEPS // 2) < TIME_STEPS // 4:
            grasp_phase = "REACHING"
        else:
            grasp_phase = "IDLE"
        
        target_object._offsets3d = ([target_x], [0], [target_z])
        
        # Update text
        time_text.set_text(f'Time: {time*1000/60:.1f}ms  |  Frame: {frame+1}/{TIME_STEPS}')
        status_text.set_text(f'Status: {grasp_phase}  |  B2Z1 Training Progress Demo')
        
        return body_line, *legs_line, *leg_feet, arm_shoulder_to_elbow, arm_elbow_to_wrist, wrist_point, target_object, time_text, status_text
    
    # Create animation
    print("Generating animation frames...")
    anim = animation.FuncAnimation(
        fig, update,
        frames=TIME_STEPS,
        interval=1000/60,  # 60 fps
        blit=False,
        repeat=True
    )
    
    return fig, anim

def main():
    print("=" * 70)
    print("B2Z1 ROBOT SIMULATION VIDEO GENERATOR")
    print("=" * 70)
    
    # Create visualization
    print("\n[1/2] Creating 3D robot animation...")
    fig, anim = create_robot_visualization()
    
    # Save animation
    output_file = r"D:\CUHK\AIMS_5790\b2z1_simulation.mp4"
    print(f"\n[2/2] Saving animation to: {output_file}")
    print("      (This may take 1-2 minutes for high quality rendering)")
    
    try:
        # Writer: either ffmpeg or pillow
        Writer = animation.writers['pillow']  # Use PIL for compatibility
        writer = Writer(fps=60, metadata=dict(artist='OpenCode'), bitrate=1800)
        anim.save(output_file, writer=writer)
        
        # Check file size
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n[OK] Animation saved successfully!")
        print(f"     File: {output_file}")
        print(f"     Size: {size_mb:.1f} MB")
        print(f"     Duration: {TIME_STEPS/60:.1f} seconds at 60 fps")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save animation: {e}")
        print("\nTrying alternative format (GIF)...")
        try:
            output_gif = output_file.replace('.mp4', '.gif')
            Writer = animation.writers['pillow']
            writer = Writer(fps=30)
            anim.save(output_gif, writer=writer)
            print(f"[OK] Saved as GIF instead: {output_gif}")
        except Exception as e2:
            print(f"[ERROR] GIF save also failed: {e2}")
    
    # Show the figure
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Close this window when animation playback finishes")
    print("2. Open: PhD_Briefing_B2Z1_Grasping_WITH_CHART.pptx")
    print("3. Add new slide or go to Slide 5")
    print("4. Insert >> Video >> b2z1_simulation.mp4 (or .gif)")
    print("5. Position and resize the video")
    print("6. Save PowerPoint")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()
