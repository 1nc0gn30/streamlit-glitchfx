import cv2
import numpy as np
import random
import glob

def advanced_datamosh_effect(frame, slice_height=10, shift_max=20, vertical_block_width=10, vertical_shift_max=15, color_shift=5):
    h, w = frame.shape[:2]

    # Horizontal slice shifting
    temp = frame.copy()
    for y in range(0, h, slice_height):
        offset = random.randint(-shift_max, shift_max)
        temp[y:y+slice_height, :] = np.roll(temp[y:y+slice_height, :], offset, axis=1)

    # Vertical block shifting
    moshed = temp.copy()
    for x in range(0, w, vertical_block_width):
        offset = random.randint(-vertical_shift_max, vertical_shift_max)
        moshed[:, x:x+vertical_block_width] = np.roll(moshed[:, x:x+vertical_block_width], offset, axis=0)

    # Color channel shifting
    b, g, r = cv2.split(moshed)
    b = np.roll(b, random.randint(-color_shift, color_shift), axis=0)
    g = np.roll(g, random.randint(-color_shift, color_shift), axis=1)
    r = np.roll(r, random.randint(-color_shift, color_shift), axis=0)
    moshed = cv2.merge([b, g, r])
    
    return moshed

def choose_video():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 files found in this folder.")
        exit()
    
    print("📹 Available videos:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    choice = input(f"Enter number (1–{len(files)}): ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(files)):
        print("❌ Invalid selection.")
        exit()
    
    return files[int(choice) - 1]

def main():
    input_video = choose_video()
    output_video = f"moshed_{input_video}"

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("❌ Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    previous_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is not None and random.random() < 0.3:
            frame = cv2.addWeighted(frame, 0.7, previous_frame, 0.3, 0)

        processed = advanced_datamosh_effect(frame)

        if random.random() < 0.1:
            out.write(processed)

        out.write(processed)
        previous_frame = processed.copy()

        cv2.imshow("Advanced Datamosh", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Glitched video saved as: {output_video}")

if __name__ == "__main__":
    main()

