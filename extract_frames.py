"""
Extract frames from Nexar dataset videos.
Usage: python extract_frames.py
"""
import os
import cv2
import pandas as pd
from tqdm import tqdm


def extract_frames(video_path, output_folder):
    """Extract all frames from a video file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    return frame_count, fps


def process_videos(video_folder, output_folder):
    """Process all videos in a folder and extract frames."""
    video_info = {}
    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")]

    with tqdm(video_files, desc=f"Processing {video_folder}", unit="video") as pbar:
        for filename in pbar:
            video_path = os.path.join(video_folder, filename)
            video_name = os.path.splitext(filename)[0]
            video_output_folder = os.path.join(output_folder, video_name)

            total_frames, fps = extract_frames(video_path, video_output_folder)
            video_info[video_name] = {"total_frames": total_frames, "fps": fps}

            pbar.set_postfix({"video": video_name, "frames": total_frames})

    return video_info


if __name__ == "__main__":
    # Define paths
    nexar_root = "data/nexar-collision-prediction"
    train_video_folder = os.path.join(nexar_root, "train")
    test_video_folder = os.path.join(nexar_root, "test")
    train_output = os.path.join(nexar_root, "train_raw_frames")
    test_output = os.path.join(nexar_root, "test_raw_frames")

    print("=" * 60)
    print("Nexar Dataset Frame Extraction")
    print("=" * 60)

    # Extract training frames
    if os.path.exists(train_video_folder):
        print("\n[1/2] Extracting training video frames...")
        train_info = process_videos(train_video_folder, train_output)
        print(f"✓ Extracted frames from {len(train_info)} training videos")
    else:
        print(f"⚠ Training folder not found: {train_video_folder}")

    # Extract test frames
    if os.path.exists(test_video_folder):
        print("\n[2/2] Extracting test video frames...")
        test_info = process_videos(test_video_folder, test_output)
        print(f"✓ Extracted frames from {len(test_info)} test videos")
    else:
        print(f"⚠ Test folder not found: {test_video_folder}")

    print("\n" + "=" * 60)
    print("Frame extraction completed!")
    print("=" * 60)
