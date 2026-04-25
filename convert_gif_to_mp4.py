#!/usr/bin/env python3
"""Convert GIF to MP4 using OpenCV"""

import cv2
from PIL import Image
import os

def gif_to_mp4(gif_path, mp4_path, fps=60):
    """Convert animated GIF to MP4"""
    
    print(f"Converting GIF to MP4...")
    print(f"  Input:  {gif_path}")
    print(f"  Output: {mp4_path}")
    
    # Open GIF
    gif = Image.open(gif_path)
    
    # Get video properties
    frames = []
    durations = []
    
    try:
        while True:
            frame_rgb = gif.convert('RGB')
            frame_cv = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
            frames.append(frame_cv)
            durations.append(gif.info.get('duration', 100))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    if not frames:
        print("[ERROR] No frames in GIF")
        return False
    
    print(f"  Frames: {len(frames)}")
    print(f"  Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    
    # Write frames
    for frame in frames:
        out.write(frame)
    
    out.release()
    
    size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f"  [OK] MP4 created: {size_mb:.1f} MB")
    
    return True

if __name__ == "__main__":
    import numpy as np
    
    gif_path = r"D:\CUHK\AIMS_5790\b2z1_simulation.gif"
    mp4_path = r"D:\CUHK\AIMS_5790\b2z1_simulation.mp4"
    
    if os.path.exists(gif_path):
        gif_to_mp4(gif_path, mp4_path, fps=30)
    else:
        print(f"GIF not found: {gif_path}")
