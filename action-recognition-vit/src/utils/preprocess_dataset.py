import os
import shutil
import cv2

def augment_frames(video_path, clip_size, frame_rate):
    """
    Augment frames to ensure the folder has at least `clip_size` frames.
    Args:
        video_path (str): Path to the folder containing video frames.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    """
    all_frames = sorted(os.listdir(video_path))
    sampled_frames = all_frames[::frame_rate]

    if len(sampled_frames) >= clip_size:
        print(f"Skipping augmentation for {video_path}: Already has {len(sampled_frames)} frames.")
        return sampled_frames  # No augmentation needed

    print(f"Augmenting frames for {video_path}: Expected {clip_size}, found {len(sampled_frames)}.")

    # Duplicate frames to reach the required clip size
    while len(sampled_frames) < clip_size:
        sampled_frames.append(sampled_frames[-1])  # Duplicate the last frame

    # Save augmented frames back to the folder
    for frame in sampled_frames[len(all_frames):]:  # Only save the newly added frames
        src_path = os.path.join(video_path, frame)
        dst_path = os.path.join(video_path, f"temp_{len(all_frames):04d}.jpg")
        shutil.copy(src_path, dst_path)
        all_frames.append(f"temp_{len(all_frames):04d}.jpg")

    # Rename all frames in the folder sequentially
    all_frames = sorted(os.listdir(video_path))  # Reload all frames after augmentation
    for i, frame in enumerate(all_frames):
        src_path = os.path.join(video_path, frame)
        dst_path = os.path.join(video_path, f"{i:04d}.jpg")
        os.rename(src_path, dst_path)

    print(f"Renamed all frames in {video_path} to sequential numbering.")
    return sampled_frames

def preprocess_dataset(root_dir, clip_size=8, frame_rate=32):
    """
    Preprocess the dataset by ensuring each folder has at least `clip_size` frames.
    Args:
        root_dir (str): Path to the dataset root directory.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames.
    """
    subfolders = os.listdir(root_dir)
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for video_folder in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_folder)
            if os.path.isdir(video_path):
                all_frames = sorted(os.listdir(video_path))
                sampled_frames = all_frames[::frame_rate]

                # Skip folders that already have enough frames
                if len(sampled_frames) >= clip_size:
                    print(f"Skipping {video_path}: Already has {len(sampled_frames)} frames.")
                    continue

                # Use augmentation to ensure the folder has enough frames
                augment_frames(video_path, clip_size, frame_rate)

                # Verification: Check the total number of frames after preprocessing
                total_frames = len(os.listdir(video_path))
                print(f"Folder: {video_path}, Total Frames After Preprocessing: {total_frames}")

if __name__ == "__main__":
    # Update the dataset path as needed
    dataset_path = "/user/HS402/zs00774/Downloads/HMDB_simp"
    preprocess_dataset(dataset_path)
