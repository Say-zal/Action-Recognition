import os
import shutil
import cv2
import numpy as np

def augment_frames(video_path, clip_size, frame_rate):
    """
    Sample and augment frames to ensure at least `clip_size` frames, saving to a new folder with '_1' suffix.
    Args:
        video_path (str): Path to the folder containing video frames.
        clip_size (int): Minimum number of frames required for a valid clip.
        frame_rate (int): Sampling rate for frames (e.g., every 32nd frame).
    Returns:
        List of frame filenames in the new folder.
    """
    # Create output folder with '_1' suffix
    video_folder_name = os.path.basename(video_path)
    output_dir = os.path.join(os.path.dirname(video_path), f"{video_folder_name}_1")
    os.makedirs(output_dir, exist_ok=True)
    
    all_frames = sorted(os.listdir(video_path))
    sampled_frames = all_frames[::frame_rate]  # Sample every 32nd frame

    if len(sampled_frames) >= clip_size:
        print(f"Skipping augmentation for {video_path}: Already has {len(sampled_frames)} frames.")
        # Copy up to clip_size frames to output folder
        for i, frame in enumerate(sampled_frames[:clip_size]):
            src_path = os.path.join(video_path, frame)
            dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
            shutil.copy(src_path, dst_path)
        return [f"{i:04d}.jpg" for i in range(len(sampled_frames[:clip_size]))]

    print(f"Augmenting frames for {video_path}: Expected {clip_size}, found {len(sampled_frames)}.")
    
    # Augment by repeating the frame sequence
    augmented_frames = []
    while len(augmented_frames) < clip_size:
        augmented_frames.extend(sampled_frames)
    augmented_frames = augmented_frames[:clip_size]  # Take exactly clip_size frames
    
    # Save augmented frames to output folder
    for i, frame in enumerate(augmented_frames):
        src_path = os.path.join(video_path, frame)
        dst_path = os.path.join(output_dir, f"{i:04d}.jpg")
        shutil.copy(src_path, dst_path)
    
    print(f"Saved {len(augmented_frames)} frames to {output_dir} with sequential numbering.")
    return [f"{i:04d}.jpg" for i in range(len(augmented_frames))]

def preprocess_dataset(root_dir, clip_size=8, frame_rate=32):
    """
    Preprocess the dataset by sampling/augmenting frames and saving to new '_1' folders, then delete original folders.
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
        video_folders = sorted(os.listdir(subfolder_path))
        for video_folder in video_folders:
            video_path = os.path.join(subfolder_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            all_frames = sorted(os.listdir(video_path))
            sampled_frames = all_frames[::frame_rate]

            # Process frames and save to new '_1' folder
            augment_frames(video_path, clip_size, frame_rate)
            
            # Verify the new folder
            new_folder = os.path.join(subfolder_path, f"{video_folder}_1")
            total_frames = len(os.listdir(new_folder)) if os.path.exists(new_folder) else 0
            print(f"Folder: {new_folder}, Total Frames After Preprocessing: {total_frames}")
            
            # Delete the original folder
            try:
                shutil.rmtree(video_path)
                print(f"Deleted original folder: {video_path}")
            except Exception as e:
                print(f"Error deleting {video_path}: {e}")

if __name__ == "__main__":
    dataset_path = "/user/HS402/zs00774/Downloads/HMDB_simp"
    preprocess_dataset(dataset_path)
