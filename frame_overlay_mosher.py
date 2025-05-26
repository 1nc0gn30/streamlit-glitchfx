import cv2
import numpy as np
import random
import moviepy.editor as mp
import os
import glob

def add_glitch_effect(frame):
    height, width, _ = frame.shape
    effect_type = random.choice(["rgb_shift", "static", "datamosh"])

    if effect_type == "rgb_shift":
        b, g, r = cv2.split(frame)
        shift_x, shift_y = random.randint(-20, 20), random.randint(-20, 20)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        r = cv2.warpAffine(r, M, (width, height))
        frame = cv2.merge([b, g, r])

    elif effect_type == "static":
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

    elif effect_type == "datamosh":
        x1, x2 = sorted(random.sample(range(width), 2))
        y1, y2 = sorted(random.sample(range(height), 2))
        frame[y1:y2, x1:x2] = np.sort(frame[y1:y2, x1:x2], axis=0)

    return frame

def overlay_random_frames(frames, width, height):
    num_frames = len(frames)
    output_frames = []

    for i, frame in enumerate(frames):
        if random.random() < 0.3:
            overlay_frame = frames[random.randint(0, num_frames - 1)]
            size = random.choice(["small", "medium", "large"])

            if size == "small":
                overlay_resized = cv2.resize(overlay_frame, (width // 4, height // 4))
                x_offset = random.randint(0, width - width // 4)
                y_offset = random.randint(0, height - height // 4)
            elif size == "medium":
                overlay_resized = cv2.resize(overlay_frame, (width // 2, height // 2))
                x_offset = random.randint(0, width - width // 2)
                y_offset = random.randint(0, height - height // 2)
            else:
                overlay_resized = cv2.resize(overlay_frame, (width, height))
                x_offset, y_offset = 0, 0

            frame[y_offset:y_offset+overlay_resized.shape[0],
                  x_offset:x_offset+overlay_resized.shape[1]] = overlay_resized

        output_frames.append(frame)

    return output_frames

def add_transitions(input_file, output_file):
    clip = mp.VideoFileClip(input_file)
    clips = []
    duration = clip.duration
    chunk_size = random.uniform(0.5, 2.0)

    start_time = 0
    while start_time < duration:
        end_time = min(start_time + chunk_size, duration)
        subclip = clip.subclip(start_time, end_time)

        effect_type = random.choice(["invert", "static_trail", "datamosh_warp"])

        if effect_type == "invert":
            subclip = subclip.fl_image(lambda frame: cv2.bitwise_not(frame))
        elif effect_type == "static_trail":
            subclip = subclip.fx(mp.vfx.lum_contrast, lum=2.0, contrast=2.0).fx(mp.vfx.time_mirror)
        elif effect_type == "datamosh_warp":
            subclip = subclip.fx(mp.vfx.speedx, factor=random.uniform(0.8, 1.2))

        clips.append(subclip)
        start_time += chunk_size

    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_file, codec="libx264", fps=clip.fps)

def choose_video():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 videos found.")
        exit()

    print("📂 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selection = input("Select a video number to glitch: ").strip()
    if not selection.isdigit() or not (1 <= int(selection) <= len(files)):
        print("❌ Invalid selection.")
        exit()

    return files[int(selection) - 1]

def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if random.random() < 0.3:
            frame = add_glitch_effect(frame)
        frames.append(frame)
    cap.release()

    frames = overlay_random_frames(frames, width, height)

    temp_file = "temp_glitched.mp4"
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    add_transitions(temp_file, output_file)
    os.remove(temp_file)

def main():
    input_video = choose_video()
    output_video = f"glitched_{os.path.splitext(os.path.basename(input_video))[0]}.mp4"
    process_video(input_video, output_video)
    print(f"✅ Glitched video saved as {output_video}")

if __name__ == "__main__":
    main()

