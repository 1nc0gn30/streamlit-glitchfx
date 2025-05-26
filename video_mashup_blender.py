import cv2
import numpy as np
import random
import glob
import os

def advanced_mash_effect(frame1, frame2, slice_height=10, shift_max=20, static_intensity=30, blend_probability=0.5):
    r = random.random()
    if r < blend_probability:
        mash_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
    elif r < (blend_probability + (1 - blend_probability) / 2):
        mash_frame = frame1.copy()
    else:
        mash_frame = frame2.copy()
        
    h, w = mash_frame.shape[:2]

    static_noise = np.random.randint(0, static_intensity, (h, w, 3), dtype=np.uint8)
    mash_frame = cv2.add(mash_frame, static_noise)

    output = mash_frame.copy()
    for y in range(0, h, slice_height):
        offset = random.randint(-shift_max, shift_max)
        output[y:y+slice_height, :] = np.roll(output[y:y+slice_height, :], offset, axis=1)

    vertical_block_width = 10
    for x in range(0, w, vertical_block_width):
        offset = random.randint(-shift_max, shift_max)
        output[:, x:x+vertical_block_width] = np.roll(output[:, x:x+vertical_block_width], offset, axis=0)

    return output

def choose_two_videos():
    files = glob.glob("*.mp4")
    if len(files) < 2:
        print("❌ Need at least two MP4 videos.")
        exit()

    print("🎬 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")

    indices = input("Enter two video numbers (comma-separated): ").split(",")
    if len(indices) != 2 or not all(i.strip().isdigit() and 1 <= int(i.strip()) <= len(files) for i in indices):
        print("❌ Invalid selection. You must select two valid numbers.")
        exit()

    idx1, idx2 = int(indices[0]) - 1, int(indices[1]) - 1
    return files[idx1], files[idx2]

def main():
    video1_path, video2_path = choose_two_videos()
    output_video_path = f"mashed_{os.path.splitext(os.path.basename(video1_path))[0]}_{os.path.splitext(os.path.basename(video2_path))[0]}.mp4"
    
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error opening one of the video files.")
        return

    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame2 = cv2.resize(frame2, (width, height))
        processed_frame = advanced_mash_effect(frame1, frame2)
        out.write(processed_frame)

        cv2.imshow("🌀 Video Mash", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Mashed video saved as: {output_video_path}")

if __name__ == '__main__':
    main()

