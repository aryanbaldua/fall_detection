"""
pose_estimation.py
------------------
Uses MediaPipe Pose Landmarker to detect 33 body keypoints (landmarks) on a
person in real-time from the webcam.

We draw the skeleton MANUALLY — no built-in visualizer.
That means drawing each landmark as a circle and each bone as a line
using raw coordinate data. This forces us to understand the data structure.

On first run this script downloads the pose landmarker model (~6MB).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# --- Download the model if we don't have it yet ---
# MediaPipe 0.10+ uses a .task file (a packaged neural network model).
# This downloads once and is reused every run after that.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pose_landmarker_full.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}.")
    print("Download it with:")
    print(f'  curl -L "{MODEL_URL}" -o {MODEL_PATH}')
    exit(1)

# --- Set up the pose landmarker ---
# BaseOptions points to the model file on disk.
# PoseLandmarkerOptions configures how the model runs.
# RunningMode.IMAGE means we pass one frame at a time and get results back immediately.
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# --- Skeleton connections ---
# This is the list of (index_a, index_b) pairs that define which landmarks
# are connected by a "bone". We define it manually here based on the
# MediaPipe Pose landmark map (33 landmarks total).
# See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Visibility threshold — skip landmarks the model isn't confident about
VISIBILITY_THRESHOLD = 0.5


def draw_skeleton(frame, landmarks, frame_width, frame_height):
    """
    Draws circles for each landmark and lines for each connection.

    landmarks    — list of 33 NormalizedLandmark objects from MediaPipe
    frame_width  — pixel width of the frame
    frame_height — pixel height of the frame
    """

    # --- Convert all 33 landmarks to pixel coordinates ---
    # MediaPipe gives normalized x, y (0.0–1.0 relative to frame size).
    # Multiply by pixel dimensions to get actual screen positions.
    pixel_coords = []
    for lm in landmarks:
        px = int(lm.x * frame_width)
        py = int(lm.y * frame_height)
        pixel_coords.append((px, py, lm.visibility))

    # --- Draw each landmark as a filled circle ---
    for px, py, visibility in pixel_coords:
        if visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)  # yellow dot

    # --- Draw each bone as a line between two landmarks ---
    for index_a, index_b in CONNECTIONS:
        px_a, py_a, vis_a = pixel_coords[index_a]
        px_b, py_b, vis_b = pixel_coords[index_b]

        # Only draw if both endpoints are visible
        if vis_a < VISIBILITY_THRESHOLD or vis_b < VISIBILITY_THRESHOLD:
            continue

        cv2.line(frame, (px_a, py_a), (px_b, py_b), (0, 255, 0), 2)  # green line


def run_pose_estimation():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {frame_width}x{frame_height}")
    print("Warming up camera...")
    for _ in range(10):   # discard first 10 frames while camera adjusts exposure
        cap.read()
    print("Press 'q' to quit.")

    first_detection_printed = False  # so we only print landmark data once

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # --- Convert BGR → RGB ---
        # OpenCV captures in BGR. MediaPipe expects RGB.
        # Skipping this causes color swapping and worse detection.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Wrap the frame in a MediaPipe Image object ---
        # The new Tasks API requires frames to be wrapped in mp.Image
        # rather than passed as raw numpy arrays.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- Run pose detection ---
        result = landmarker.detect(mp_image)

        # result.pose_landmarks is a list of detected people.
        # Each person is a list of 33 landmarks.
        # We only care about the first detected person: result.pose_landmarks[0]
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]  # first person's 33 landmarks

            draw_skeleton(frame, landmarks, frame_width, frame_height)

            # Print landmark data once so we can inspect the structure
            if not first_detection_printed:
                lm = landmarks[0]  # landmark 0 = nose
                print(f"\nExample landmark (nose):")
                print(f"  x (normalized): {lm.x:.4f}")
                print(f"  y (normalized): {lm.y:.4f}")
                print(f"  z (depth):      {lm.z:.4f}")
                print(f"  visibility:     {lm.visibility:.4f}")
                first_detection_printed = True

        else:
            cv2.putText(frame, "No person detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_pose_estimation()
