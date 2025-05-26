import cv2
import numpy as np
import random
import glob

# ---------- Datamosh Effects ----------

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

# ---------- Fake Object Detection Overlay ----------

def fake_detection_overlay(frame, detection_probability=0.3):
    if random.random() < detection_probability:
        h, w = frame.shape[:2]
        num_boxes = random.randint(1, 5)
        for _ in range(num_boxes):
            box_w = random.randint(30, w // 4)
            box_h = random.randint(30, h // 4)
            x1 = random.randint(0, w - box_w)
            y1 = random.randint(0, h - box_h)
            x2 = x1 + box_w
            y2 = y1 + box_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"obj {random.randint(0, 99)}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ---------- Transition Effect ----------

def fade_distortion_transition(frame1, frame2, num_frames=15):
    transition_frames = []
    for i in range(num_frames):
        alpha = i / float(num_frames)
        blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        blended = datamosh_effect_1(blended, slice_height=5, shift_max=5)
        transition_frames.append(blended)
    return transition_frames

# ---------- Clip Extraction ----------

def extract_clip(file, duration_seconds=3, target_resolution=None):
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print(f"❌ Failed to open {file}")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(round(fps))
    total_frames_needed = int(duration_seconds * fps_int)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = random.randint(0, max(0, total_frames - total_frames_needed))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    clip_frames = []
    for _ in range(total_frames_needed):
        ret, frame = cap.read()
        if not ret:
            break
        if target_resolution:
            frame = cv2.resize(frame, target_resolution)
        frame = datamosh_effect_1(frame)
        frame = fake_detection_overlay(frame)
        clip_frames.append(frame)

    cap.release()
    return clip_frames, fps_int

# ---------- File Chooser ----------

def choose_videos():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 files found.")
        exit()

    print("🎬 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    selected = input("Enter video numbers to include (comma-separated): ").split(",")
    chosen = []
    for s in selected:
        s = s.strip()
        if s.isdigit() and 1 <= int(s) <= len(files):
            chosen.append(files[int(s) - 1])

    if not chosen:
        print("❌ No valid selections made.")
        exit()

    return chosen

# ---------- Main Assembly ----------

def main():
    selected_files = choose_videos()

    final_frames = []
    total_duration_seconds = 60
    target_clip_duration = 3
    transition_duration_frames = 15

    first_clip, fps = extract_clip(selected_files[0], duration_seconds=target_clip_duration)
    if not first_clip:
        print("❌ Could not extract a clip from the first video.")
        return
    target_resolution = (first_clip[0].shape[1], first_clip[0].shape[0])
    final_frames.extend(first_clip)
    previous_clip_last_frame = first_clip[-1]

    total_frames_needed = total_duration_seconds * fps
    current_frame_count = len(final_frames)
    file_index = 1
    while current_frame_count < total_frames_needed:
        file_path = selected_files[file_index % len(selected_files)]
        clip, _ = extract_clip(file_path, duration_seconds=target_clip_duration, target_resolution=target_resolution)
        if clip:
            transition = fade_distortion_transition(previous_clip_last_frame, clip[0], num_frames=transition_duration_frames)
            final_frames.extend(transition)
            final_frames.extend(clip)
            previous_clip_last_frame = clip[-1]
            current_frame_count = len(final_frames)
        file_index += 1

    final_frames = final_frames[:total_frames_needed]
    final_frames = [datamosh_effect_2(frame) for frame in final_frames]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("final_minute_moshed.mp4", fourcc, fps, target_resolution)
    for frame in final_frames:
        out.write(frame)
    out.release()

    print("✅ Final 1-minute datamoshed video saved as: final_minute_moshed.mp4")

if __name__ == '__main__':
    main()

