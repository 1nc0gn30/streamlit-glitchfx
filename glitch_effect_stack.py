import cv2
import numpy as np
import random
import moviepy.editor as mp
import os
import glob

def add_glitch_effect(frame):
    height, width, channels = frame.shape
    glitch_type = random.choice(["rgb_shift", "noise", "pixel_sort"])
    
    if glitch_type == "rgb_shift":
        b, g, r = cv2.split(frame)
        shift_x = random.randint(-10, 10)
        shift_y = random.randint(-10, 10)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        r = cv2.warpAffine(r, M, (width, height))
        frame = cv2.merge([b, g, r])

    elif glitch_type == "noise":
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

    elif glitch_type == "pixel_sort":
        x1, x2 = sorted(random.sample(range(width), 2))
        y1, y2 = sorted(random.sample(range(height), 2))
        frame[y1:y2, x1:x2] = np.sort(frame[y1:y2, x1:x2], axis=0)

    return frame

def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("temp_glitched.mp4", fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if random.random() < 0.2:
            frame = add_glitch_effect(frame)

        if random.random() < 0.05:
            for _ in range(random.randint(2, 5)):
                out.write(frame)

        out.write(frame)

    cap.release()
    out.release()

    add_transitions("temp_glitched.mp4", output_file)
    os.remove("temp_glitched.mp4")

def add_transitions(input_file, output_file):
    clip = mp.VideoFileClip(input_file)
    clips = []
    duration = clip.duration
    chunk_size = random.uniform(0.5, 2.0)

    start_time = 0
    while start_time < duration:
        end_time = min(start_time + chunk_size, duration)
        subclip = clip.subclip(start_time, end_time)

        effect_type = random.choice(["invert", "contrast", "warp"])

        if effect_type == "invert":
            subclip = subclip.fl_image(lambda frame: cv2.bitwise_not(frame))
        elif effect_type == "contrast":
            subclip = subclip.fl_image(lambda frame: cv2.convertScaleAbs(frame, alpha=1.5, beta=20))
        elif effect_type == "warp":
            subclip = subclip.fx(mp.vfx.lum_contrast, lum=1.5, contrast=2.0)

        clips.append(subclip)
        start_time += chunk_size

    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", fps=clip.fps)

def choose_video():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 files found.")
        exit()

    print("📂 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selection = input("Select a video number to glitch: ").strip()
    if not selection.isdigit() or not (1 <= int(selection) <= len(files)):
        print("❌ Invalid selection.")
        exit()

    return files[int(selection) - 1]

def main():
    input_video = choose_video()
    name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = f"glitchstack_{name}.mp4"
    process_video(input_video, output_video)
    print(f"✅ Glitched video saved as: {output_video}")

if __name__ == "__main__":
    main()

