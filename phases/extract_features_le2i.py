"""
extract_features_le2i.py
------------------------
Runs MediaPipe pose estimation on every video in the Le2i dataset and saves
the per-frame body_angle and hip_height values to a CSV file.

You only need to run this ONCE. After that, evaluate_le2i.py reads the CSV
directly — no MediaPipe required — so changing thresholds is instant.

Output: data/le2i_features.csv
Columns: scene, video, true_label, frame, angle, hip_height

true_label is determined from the annotation file:
  - First two lines are fall_start and fall_end frame numbers
  - Both non-zero → FALL
  - Both zero     → ADL (no fall)
  - No header (starts with CSV row) → FALL (known exceptions in this dataset)
"""

import os
import csv
import glob

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_extraction import compute_body_angle, compute_hip_height, compute_bbox_ratio, MODEL_PATH

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Le2i")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "le2i_features.csv")

SCENES = [
    ("Coffee_room_01", "Coffee_room_01/Coffee_room_01/Videos", "Coffee_room_01/Coffee_room_01/Annotation_files"),
    ("Coffee_room_02", "Coffee_room_02/Coffee_room_02/Videos", "Coffee_room_02/Coffee_room_02/Annotations_files"),
    ("Home_01",        "Home_01/Home_01/Videos",               "Home_01/Home_01/Annotation_files"),
    ("Home_02",        "Home_02/Home_02/Videos",               "Home_02/Home_02/Annotation_files"),
]


# ─────────────────────────────────────────────
# LABEL PARSING
# ─────────────────────────────────────────────

def get_label(ann_path):
    """
    Returns "FALL" or "ADL" based on the annotation file.

    If the first line is a plain integer:
      - fall_start = line 1, fall_end = line 2
      - both non-zero → FALL, both zero → ADL

    If the first line is a CSV row (no header):
      - these 3 known files are all falls
    """
    with open(ann_path) as f:
        lines = f.readlines()

    first = lines[0].strip()

    if "," not in first:
        fall_start = int(first)
        fall_end   = int(lines[1].strip())
        return "FALL" if (fall_start != 0 and fall_end != 0) else "ADL"

    # No header — known to be falls
    return "FALL"


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_video_features(video_path, landmarker):
    """
    Reads every frame in a video and returns a list of (frame_num, angle, hip_height).

    angle and hip_height are None if MediaPipe couldn't detect a pose in that frame.
    We still record those frames so the CSV has a complete frame-by-frame record.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path}")
        return []

    rows = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            rows.append((frame_num, None, None, None))
            continue

        lm         = result.pose_landmarks[0]
        angle      = compute_body_angle(lm)
        hip_height = compute_hip_height(lm)
        bbox_ratio = compute_bbox_ratio(lm)

        rows.append((frame_num, angle, hip_height, bbox_ratio))

    cap.release()
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    if not os.path.exists(MODEL_PATH):
        print(f"MediaPipe model not found at: {MODEL_PATH}")
        return

    # Load MediaPipe once — reused for every video
    print("Loading MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Open CSV for writing
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["scene", "video", "true_label", "frame", "angle", "hip_height", "bbox_ratio"])

        total_videos = 0
        total_frames = 0

        for scene_name, vid_rel, ann_rel in SCENES:
            vid_dir = os.path.join(DATA_DIR, vid_rel)
            ann_dir = os.path.join(DATA_DIR, ann_rel)

            video_paths = sorted(glob.glob(os.path.join(vid_dir, "*.avi")))

            for vpath in video_paths:
                stem     = os.path.basename(vpath).replace(".avi", "")
                ann_path = os.path.join(ann_dir, stem + ".txt")

                if not os.path.exists(ann_path):
                    print(f"  [WARNING] No annotation for {stem} in {scene_name} — skipping")
                    continue

                label = get_label(ann_path)
                print(f"  Processing [{scene_name}] {stem} ({label})...")

                frame_rows = extract_video_features(vpath, landmarker)

                for frame_num, angle, hip_height, bbox_ratio in frame_rows:
                    writer.writerow([
                        scene_name,
                        stem,
                        label,
                        frame_num,
                        f"{angle:.4f}"      if angle      is not None else "",
                        f"{hip_height:.4f}" if hip_height is not None else "",
                        f"{bbox_ratio:.4f}" if bbox_ratio is not None else "",
                    ])

                total_videos += 1
                total_frames += len(frame_rows)

    print(f"\nDone. {total_videos} videos, {total_frames} frames.")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
