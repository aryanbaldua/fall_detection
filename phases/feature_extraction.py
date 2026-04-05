"""
feature_extraction.py
---------------------
Builds on Phase 3 (pose estimation) by computing three meaningful features
from the raw keypoint positions. These features are what the fall detector
will actually use — not the raw coordinates themselves.

The three features:
  1. body_angle  — angle of the spine relative to vertical (0° = upright, 90° = horizontal)
  2. hip_height  — normalized y position of the hips (larger = lower in frame = closer to ground)
  3. velocity    — how fast the hips are moving downward between frames

The real-time overlay displays all three as live numbers so you can watch
how they change as you move, bend, crouch, or simulate a fall.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pose_landmarker_full.task")

# --- Landmark indices we care about ---
# We only need 4 landmarks to compute all three features.
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24

VISIBILITY_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# FEATURE 1: Body Angle
# ─────────────────────────────────────────────
def compute_body_angle(landmarks):
    """
    Computes the angle between the spine and the vertical axis.

    The "spine" here is an approximation: a line from the midpoint of the
    shoulders down to the midpoint of the hips.

    When upright:   spine points straight down  → angle ≈ 0°
    When fallen:    spine points sideways        → angle ≈ 90°
    When upside down: spine points up            → angle ≈ 180° (rare)

    How it works (dot product method):
    ┌─────────────────────────────────────────────────┐
    │  Two vectors:                                   │
    │    spine    = hip_mid - shoulder_mid            │
    │    vertical = (0, 1)  ← points straight down   │
    │                          in image coordinates   │
    │                                                 │
    │  cos(angle) = dot(a, b) / (|a| * |b|)          │
    │  angle      = arccos(result)                    │
    └─────────────────────────────────────────────────┘

    Remember: in image coordinates, y=0 is the TOP of the frame and
    y increases DOWNWARD. So "straight down" is the +y direction = (0, 1).
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    # Check that all 4 landmarks are visible enough to trust
    if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < VISIBILITY_THRESHOLD:
        return None  # can't compute reliably

    # Midpoint of the two shoulders (average of x and y)
    shoulder_mid = np.array([
        (ls.x + rs.x) / 2,
        (ls.y + rs.y) / 2
    ])

    # Midpoint of the two hips
    hip_mid = np.array([
        (lh.x + rh.x) / 2,
        (lh.y + rh.y) / 2
    ])

    # Spine vector: from shoulder midpoint → hip midpoint
    # When upright this points downward: roughly (0, positive)
    spine_vec = hip_mid - shoulder_mid

    # Vertical vector: straight down in image coordinates
    vertical_vec = np.array([0, 1])

    # --- Dot product to find angle ---
    # np.dot(a, b) = |a| * |b| * cos(angle)
    # Rearranging: cos(angle) = dot(a, b) / (|a| * |b|)
    # np.linalg.norm(v) = |v| = length of a vector

    spine_length = np.linalg.norm(spine_vec)
    if spine_length == 0:
        return None  # avoid division by zero if landmarks overlap

    cos_angle = np.dot(spine_vec, vertical_vec) / spine_length
    # np.clip keeps the value between -1 and 1 to avoid floating point errors
    # that could make arccos crash (e.g. 1.0000001 is invalid for arccos)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # arccos gives the angle in radians — convert to degrees
    angle_degrees = np.degrees(np.arccos(cos_angle))
    return angle_degrees


# ─────────────────────────────────────────────
# FEATURE 2: Hip Height
# ─────────────────────────────────────────────
def compute_hip_height(landmarks):
    """
    Returns the normalized y position of the hip midpoint.

    Range: 0.0 (top of frame) to 1.0 (bottom of frame)

    When standing:  hips are roughly in the middle → ~0.5–0.6
    When fallen:    hips drop toward the bottom    → closer to 1.0

    This is already normalized (0–1) by MediaPipe so it's consistent
    across different camera resolutions.
    """
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    if min(lh.visibility, rh.visibility) < VISIBILITY_THRESHOLD:
        return None

    hip_mid_y = (lh.y + rh.y) / 2
    return hip_mid_y


# ─────────────────────────────────────────────
# FEATURE 3: Velocity
# ─────────────────────────────────────────────
def compute_velocity(current_hip_y, previous_hip_y):
    """
    Returns how much the hip height changed between this frame and the last.

    velocity = current_hip_y - previous_hip_y

    Positive → hips moved DOWN  (falling, crouching, sitting)
    Negative → hips moved UP    (standing up, jumping)
    Near zero → little movement

    A fall has a brief spike of HIGH positive velocity (rapid downward movement)
    followed by near-zero (person is on the ground, not moving).
    """
    if current_hip_y is None or previous_hip_y is None:
        return None
    return current_hip_y - previous_hip_y


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

def draw_skeleton(frame, landmarks, frame_width, frame_height):
    pixel_coords = []
    for lm in landmarks:
        px = int(lm.x * frame_width)
        py = int(lm.y * frame_height)
        pixel_coords.append((px, py, lm.visibility))

    for px, py, visibility in pixel_coords:
        if visibility < VISIBILITY_THRESHOLD:
            continue
        cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

    for index_a, index_b in CONNECTIONS:
        px_a, py_a, vis_a = pixel_coords[index_a]
        px_b, py_b, vis_b = pixel_coords[index_b]
        if vis_a < VISIBILITY_THRESHOLD or vis_b < VISIBILITY_THRESHOLD:
            continue
        cv2.line(frame, (px_a, py_a), (px_b, py_b), (0, 255, 0), 2)


def draw_feature_overlay(frame, body_angle, hip_height, velocity):
    """
    Draws the three feature values as a live readout in the top-left corner.
    Color-codes body angle as a warning when it goes high (potential fall).
    """
    # --- Body angle ---
    if body_angle is not None:
        # Green when upright, yellow when leaning, red when nearly horizontal
        if body_angle < 30:
            angle_color = (0, 255, 0)    # green
        elif body_angle < 60:
            angle_color = (0, 200, 255)  # yellow
        else:
            angle_color = (0, 0, 255)    # red
        cv2.putText(frame, f"Body Angle:  {body_angle:.1f} deg",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, angle_color, 2)
    else:
        cv2.putText(frame, "Body Angle:  --",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # --- Hip height ---
    if hip_height is not None:
        cv2.putText(frame, f"Hip Height:  {hip_height:.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    else:
        cv2.putText(frame, "Hip Height:  --",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # --- Velocity ---
    if velocity is not None:
        vel_color = (0, 255, 255) if abs(velocity) < 0.01 else (0, 100, 255)
        cv2.putText(frame, f"Velocity:    {velocity:.4f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vel_color, 2)
    else:
        cv2.putText(frame, "Velocity:    --",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run_feature_extraction():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Download it with:")
        print(f'  curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o {MODEL_PATH}')
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Warming up camera...")
    for _ in range(10):
        cap.read()
    print("Press 'q' to quit.")
    print("Try: standing upright, leaning, crouching, lying down.")
    print("Watch how body_angle, hip_height, and velocity respond.\n")

    previous_hip_y = None  # stores last frame's hip height to compute velocity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect(mp_image)

        body_angle = None
        hip_height = None
        velocity   = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            draw_skeleton(frame, landmarks, frame_width, frame_height)

            # Compute all three features from raw landmark positions
            body_angle = compute_body_angle(landmarks)
            hip_height = compute_hip_height(landmarks)
            velocity   = compute_velocity(hip_height, previous_hip_y)

            # Save this frame's hip height so next frame can compute velocity
            previous_hip_y = hip_height

            # Print to terminal so you can see the raw numbers
            if body_angle is not None:
                print(f"angle={body_angle:5.1f}°  hip_y={hip_height:.3f}  vel={velocity:+.4f}" if velocity else
                      f"angle={body_angle:5.1f}°  hip_y={hip_height:.3f}  vel=--")
        else:
            previous_hip_y = None  # reset if person leaves frame
            cv2.putText(frame, "No person detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        draw_feature_overlay(frame, body_angle, hip_height, velocity)

        cv2.imshow("Feature Extraction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_feature_extraction()
