import cv2
import numpy as np
import random
import glob

def datamosh_effect(frame, slice_height=10, shift_max=20):
    """
    Applies a datamosh-like effect by slicing the frame into horizontal bands
    and randomly shifting each band horizontally.
    """
    h, w = frame.shape[:2]
    output = frame.copy()
    for y in range(0, h, slice_height):
        offset = random.randint(-shift_max, shift_max)
        output[y:y+slice_height, :] = np.roll(frame[y:y+slice_height, :], offset, axis=1)
    return output

def choose_video():
    files = glob.glob("*.mp4")
    if not files:
        print("❌ No MP4 videos found in this folder.")
        exit()

    print("📂 Select a video to glitch:")
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = datamosh_effect(frame, slice_height=10, shift_max=20)
        out.write(processed)

        cv2.imshow("Datamosh Effect", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Glitched video saved as: {output_video}")

if __name__ == '__main__':
    main()

