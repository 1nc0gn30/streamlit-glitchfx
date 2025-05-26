import cv2
import numpy as np
import random
import glob

def datamosh_effect_1(frame, slice_height=10, shift_max=20):
    h, w = frame.shape[:2]
    output = frame.copy()
    for y in range(0, h, slice_height):
        offset = random.randint(-shift_max, shift_max)
        output[y:y+slice_height, :] = np.roll(output[y:y+slice_height, :], offset, axis=1)
    return output

def datamosh_effect_2(frame, block_size=10, shift_max=15, color_shift=5):
    h, w = frame.shape[:2]
    output = frame.copy()
    for x in range(0, w, block_size):
        offset = random.randint(-shift_max, shift_max)
        output[:, x:x+block_size] = np.roll(output[:, x:x+block_size], offset, axis=0)
    b, g, r = cv2.split(output)
    b = np.roll(b, random.randint(-color_shift, color_shift), axis=1)
    g = np.roll(g, random.randint(-color_shift, color_shift), axis=0)
    r = np.roll(r, random.randint(-color_shift, color_shift), axis=1)
    return cv2.merge([b, g, r])

def process_video(file, target_resolution=None):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print(f"❌ Failed to open {file}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(round(fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < fps_int:
        print(f"⚠️  {file} is too short for a 1-second clip.")
        cap.release()
        return [], 0

    start_frame = random.randint(0, total_frames - fps_int)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    clip_frames = []
    for _ in range(fps_int):
        ret, frame = cap.read()
        if not ret:
            break
        if target_resolution is not None:
            frame = cv2.resize(frame, target_resolution)
        frame = datamosh_effect_1(frame)
        clip_frames.append(frame)

    cap.release()
    return clip_frames, fps_int

def choose_videos():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 files found.")
        exit()

    print("📂 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selections = input("Enter numbers of videos to use (comma-separated): ").split(",")
    selected = []
    for s in selections:
        s = s.strip()
        if s.isdigit() and 1 <= int(s) <= len(files):
            selected.append(files[int(s) - 1])

    if not selected:
        print("❌ No valid selections made.")
        exit()

    return selected

def main():
    selected_files = choose_videos()
    all_frames = []
    video_fps = 0
    target_resolution = None

    first_clip, video_fps = process_video(selected_files[0])
    if not first_clip:
        print("❌ No valid clip extracted from the first video.")
        return
    target_resolution = (first_clip[0].shape[1], first_clip[0].shape[0])
    all_frames.extend(first_clip)

    for file in selected_files[1:]:
        clip_frames, _ = process_video(file, target_resolution=target_resolution)
        if clip_frames:
            all_frames.extend(clip_frames)

    if not all_frames:
        print("❌ No frames extracted.")
        return

    final_frames = [datamosh_effect_2(frame) for frame in all_frames]

    output_file = "final_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, video_fps, target_resolution)

    for frame in final_frames:
        out.write(frame)
    out.release()

    print(f"✅ Final datamoshed video saved as: {output_file}")

if __name__ == '__main__':
    main()

