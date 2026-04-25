from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


PRES_DIR = Path(r"D:\CUHK\AIMS_5790\presentation")
ROOT_DIR = PRES_DIR.parent


def load_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    h, w = frames[0].shape[:2]
    return frames, fps, (w, h)


def write_video(video_path, frames, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    for frame in frames:
        writer.write(frame)
    writer.release()


def make_clip(src, dst, max_frames, loop=False):
    frames, fps, size = load_frames(src)

    if loop and len(frames) < max_frames:
        out = []
        idx = 0
        while len(out) < max_frames:
            out.append(frames[idx % len(frames)])
            idx += 1
    else:
        out = frames[:max_frames]

    write_video(dst, out, fps, size)
    return len(out), fps


def extract_frame(video_path, frame_idx, out_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame extraction: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not extract frame {frame_idx} from {video_path}")

    cv2.imwrite(str(out_path), frame)


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
                Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            ]
        )
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
        ]
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(draw, box, text, font, fill):
    left, top, right, bottom = box
    text_box = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=8)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=8)


def make_video_card(out_path, title, subtitle, duration_s):
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    purple = (112, 48, 160)
    light = (245, 242, 250)
    grey = (95, 95, 95)
    dark = (35, 35, 35)

    draw.rounded_rectangle((50, 50, w - 50, h - 50), radius=28, fill=light, outline=purple, width=6)
    draw.rounded_rectangle((90, 90, w - 90, h - 90), radius=24, fill=(255, 255, 255), outline=(220, 220, 220), width=2)

    cx, cy = w // 2, 285
    r = 82
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=purple)
    draw.polygon([(cx - 18, cy - 34), (cx - 18, cy + 34), (cx + 42, cy)], fill=(255, 255, 255))

    title_font = load_font(46, bold=True)
    subtitle_font = load_font(28, bold=False)
    footer_font = load_font(26, bold=False)

    draw_centered(draw, (120, 390, w - 120, 500), title, title_font, dark)
    draw_centered(draw, (120, 500, w - 120, 585), subtitle, subtitle_font, grey)
    draw_centered(draw, (120, 600, w - 120, 660), f"Click to play embedded video ({duration_s}s)", footer_font, purple)

    img.save(out_path)


def main():
    # Presentation-ready clips
    fixedbase_src = ROOT_DIR / "b2z1_fixedbase_arm.mp4"
    standing_src = ROOT_DIR / "b2z1_stand_fixed_kp200kd20.mp4"
    grasp_src = PRES_DIR / "grasp_demo_v10.mp4"

    fixedbase_dst = PRES_DIR / "fixedbase_arm.mp4"
    standing_dst = PRES_DIR / "standing_demo.mp4"
    grasp_dst = PRES_DIR / "grasp_demo.mp4"

    fixed_frames, fixed_fps = make_clip(fixedbase_src, fixedbase_dst, max_frames=240, loop=False)
    stand_frames, stand_fps = make_clip(standing_src, standing_dst, max_frames=600, loop=False)
    grasp_frames, grasp_fps = make_clip(grasp_src, grasp_dst, max_frames=600, loop=False)

    # Generic video cards so the slide never shows an outdated screenshot.
    make_video_card(
        PRES_DIR / "video_card_fixedbase.png",
        "Embedded Video",
        "Stage 1: Fixed-Base Arm Reaching",
        round(fixed_frames / fixed_fps),
    )
    make_video_card(
        PRES_DIR / "video_card_standing.png",
        "Embedded Video",
        "Demo 1: Standing Stability",
        round(stand_frames / stand_fps),
    )
    make_video_card(
        PRES_DIR / "video_card_grasp.png",
        "Embedded Video",
        "Demo 2: Arm Approaching Target Cube",
        round(grasp_frames / grasp_fps),
    )

    # Still frames used on non-video slides.
    standing_frames = [0, 120, 240, 360]
    standing_outputs = [
        PRES_DIR / "stand_fixed_f000.png",
        PRES_DIR / "stand_fixed_f100.png",
        PRES_DIR / "stand_fixed_f300.png",
        PRES_DIR / "stand_fixed_f500.png",
    ]
    for frame_idx, out_path in zip(standing_frames, standing_outputs):
        extract_frame(standing_dst, frame_idx, out_path)

    grasp_frames_idx = [0, 320, 560]
    grasp_outputs = [
        PRES_DIR / "demo_frame0.png",
        PRES_DIR / "demo_reach.png",
        PRES_DIR / "demo_grasp.png",
    ]
    for frame_idx, out_path in zip(grasp_frames_idx, grasp_outputs):
        extract_frame(grasp_dst, frame_idx, out_path)

    print(f"Prepared: {fixedbase_dst.name}, {standing_dst.name}, {grasp_dst.name}")
    print("Prepared poster cards and still frames for presentation slides")


if __name__ == "__main__":
    main()
