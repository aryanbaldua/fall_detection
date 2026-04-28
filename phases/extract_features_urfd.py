"""
extract_features_urfd.py
------------------------
Runs MediaPipe pose estimation on every video in the UR Fall Detection
Dataset (URFD) and saves per-frame body_angle, hip_height, and bbox_ratio
to a CSV.

Mirrors extract_features_le2i.py — same feature functions, same CSV columns
— so the same rule-based evaluator can be reused.

Output: data/urfd_features.csv
Columns: scene, video, true_label, frame, angle, hip_height, bbox_ratio

Label rule (video-level, matching the Le2i convention):
  - filename starts with "fall-" → true_label = "FALL"
  - filename starts with "adl-"  → true_label = "ADL"

⚠ URFD frame format note:
  Each frame in URFD's cam0 mp4s is 640x240 — actually a side-by-side
  composite of:
    LEFT half  (x =   0..319) → depth image (just a visualization)
    RIGHT half (x = 320..639) → the actual RGB camera frame
  We crop to the right half before running MediaPipe; otherwise the
  depth silhouette confuses pose detection and we lose ~half the frames.

Why we don't use URFD's per-frame labels:
  URFD ships frame-level labels (-1=upright, 0=transition, 1=on-ground),
  but our Le2i pipeline uses video-level labels. To keep the schema
  identical we use video-level here too. We can revisit per-frame later
  for the LSTM phase, where finer-grained labels actually help.
"""

import os
import csv
import glob

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Reuse the exact same feature functions used for Le2i — that is the whole
# point of this script. If those change later, both datasets stay aligned.
from feature_extraction import (
    compute_body_angle,
    compute_hip_height,
    compute_bbox_ratio,
    MODEL_PATH,
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(SCRIPT_DIR, "..", "data", "URFD", "Videos")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "..", "data", "urfd_features.csv")


# ─────────────────────────────────────────────
# LABEL + ID HELPERS
# ─────────────────────────────────────────────

def parse_filename(filename):
    """
    Turns a URFD filename into (scene, video_id, true_label).

    Input examples:
      "fall-01-cam0.mp4" → ("URFD_fall", "fall-01", "FALL")
      "adl-17-cam0.mp4"  → ("URFD_adl",  "adl-17",  "ADL")

    The scene name distinguishes fall vs ADL so plots can group by it later
    (this matches how Le2i uses scene = "Coffee_room_01" etc.).
    """
    # Strip the extension so we can work with the bare stem.
    stem = filename.replace(".mp4", "")          # e.g. "fall-01-cam0"

    # The stem is always "<kind>-<NN>-<camera>". Splitting on "-" gives 3 parts.
    parts = stem.split("-")                       # ["fall", "01", "cam0"]
    kind  = parts[0]                              # "fall" or "adl"

    # video_id keeps the kind + index, dropping the camera suffix.
    # We do this because all our videos here are cam0 — no need to record it.
    video_id = f"{kind}-{parts[1]}"               # "fall-01"

    if kind == "fall":
        return "URFD_fall", video_id, "FALL"
    return "URFD_adl", video_id, "ADL"


# ─────────────────────────────────────────────
# FEATURE EXTRACTION (per video)
# ─────────────────────────────────────────────

def extract_video_features(video_path, landmarker):
    """
    Reads every frame in a video and returns a list of
    (frame_num, angle, hip_height, bbox_ratio) tuples.

    If MediaPipe can't find a pose in a frame, the three feature values are
    None — but we still emit the row so the CSV has a complete frame index.
    Downstream code can decide how to handle missing detections.
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
            # ret=False means we've hit the end of the video (or a read error).
            break
        frame_num += 1

        # Crop to the right half — that's the actual RGB image.
        # The left half is a depth visualization that confuses MediaPipe.
        # frame.shape is (height, width, channels) → slice columns w//2 onward.
        h, w = frame.shape[:2]
        frame = frame[:, w // 2:]

        # MediaPipe expects RGB; OpenCV gives us BGR by default.
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Wrap the numpy array in MediaPipe's image container.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        # No pose detected — record a blank row and move on.
        if not result.pose_landmarks:
            rows.append((frame_num, None, None, None))
            continue

        # result.pose_landmarks is a list (one entry per detected person).
        # We assume a single subject, so take index 0.
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

    if not os.path.isdir(VIDEO_DIR):
        print(f"URFD video directory not found: {VIDEO_DIR}")
        print("Run phases/download_urfd.py first.")
        return

    # Build the MediaPipe landmarker once — it's expensive to initialize.
    print("Loading MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # sorted() so falls come before adls alphabetically — predictable order.
    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    if not video_paths:
        print(f"No mp4 files found in {VIDEO_DIR}")
        return

    print(f"Found {len(video_paths)} videos. Writing to: {OUTPUT_CSV}\n")

    total_videos = 0
    total_frames = 0

    # Open the CSV once, write a header, then stream rows as we go. This is
    # memory-friendly — we never hold the full dataset in RAM.
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["scene", "video", "true_label", "frame", "angle", "hip_height", "bbox_ratio"])

        for vpath in video_paths:
            filename = os.path.basename(vpath)
            scene, video_id, label = parse_filename(filename)

            print(f"  Processing [{scene}] {video_id} ({label})...")

            frame_rows = extract_video_features(vpath, landmarker)

            for frame_num, angle, hip_height, bbox_ratio in frame_rows:
                # Format floats to 4 decimals for compactness; keep blanks
                # when MediaPipe missed the pose.
                writer.writerow([
                    scene,
                    video_id,
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
