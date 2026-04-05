"""
rule_based_detector.py
----------------------
Phase 5: Rule-Based Fall Detector

Builds on Phase 4 (feature extraction) by adding fall detection logic.
No machine learning — just if/else rules on the three features we computed:
  - body_angle  (spine angle relative to vertical)
  - hip_height  (normalized y position of hips)
  - velocity    (downward speed of hips)

Imports the feature computation and skeleton drawing functions directly
from feature_extraction.py — no duplication.

The detection logic:
  1. Check if body_angle AND hip_height cross their thresholds simultaneously
  2. Only trigger an alert if this holds for N consecutive frames (duration filter)
  3. Once triggered, keep the alert on screen for a cooldown period

After running, deliberately try to break it:
  - Bend over to pick something up         → does it false-positive?
  - Sit down quickly                        → does it false-positive?
  - Crouch low                              → does it false-positive?
  - Lie down slowly vs fall quickly         → does it catch both?
These failure modes are exactly why we build the LSTM in Phase 6.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Import all feature functions and drawing helpers from Phase 4
from feature_extraction import (
    compute_body_angle,
    compute_hip_height,
    compute_velocity,
    CONNECTIONS,
    VISIBILITY_THRESHOLD,
    MODEL_PATH,
)

# ─────────────────────────────────────────────
# THRESHOLDS — tune these to reduce false positives/negatives
# ─────────────────────────────────────────────

# Body angle above this = significantly non-upright
# 0° = perfectly upright, 90° = horizontal (lying down)
# 45° catches falls while ignoring mild leaning
ANGLE_THRESHOLD = 45.0

# Hip height above this = hips are low in frame (normalized 0–1)
# 0.65 means hips are in the bottom 35% of the frame
HIP_HEIGHT_THRESHOLD = 0.65

# How many consecutive frames both conditions must hold before alerting
# At ~30fps: 10 frames ≈ 0.33 seconds
REQUIRED_CONSECUTIVE_FRAMES = 10

# How many frames to keep the "FALL DETECTED" alert on screen after triggering
# 90 frames ≈ 3 seconds at 30fps
ALERT_COOLDOWN_FRAMES = 90


# ─────────────────────────────────────────────
# FALL DETECTION LOGIC
# ─────────────────────────────────────────────
def check_fall_conditions(body_angle, hip_height):
    """
    Returns True if both fall conditions are met simultaneously.

    We need BOTH conditions because:
      - angle alone: bending over to pick something up = false positive
      - hip_height alone: crouching = false positive
      - BOTH together: much more specific to an actual fall
    """
    if body_angle is None or hip_height is None:
        return False

    return body_angle > ANGLE_THRESHOLD and hip_height > HIP_HEIGHT_THRESHOLD


# ─────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────
def draw_overlay(frame, body_angle, hip_height, velocity,
                 consecutive_count, fall_detected):
    """
    Draws:
      1. Feature readout (angle, hip height, velocity) — top left
      2. Consecutive frame counter — how close we are to triggering
      3. "FALL DETECTED" alert — center of frame (when triggered)
    """
    angle_color = (0, 255, 0) if (body_angle or 0) < ANGLE_THRESHOLD else (0, 0, 255)
    hip_color   = (0, 255, 0) if (hip_height or 0) < HIP_HEIGHT_THRESHOLD else (0, 0, 255)

    cv2.putText(frame,
                f"Body Angle: {body_angle:.1f} deg" if body_angle is not None else "Body Angle: --",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2)

    cv2.putText(frame,
                f"Hip Height: {hip_height:.3f}" if hip_height is not None else "Hip Height: --",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hip_color, 2)

    cv2.putText(frame,
                f"Velocity:   {velocity:+.4f}" if velocity is not None else "Velocity:   --",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Consecutive frame counter — acts like a charging bar toward triggering
    cv2.putText(frame,
                f"Trigger: {consecutive_count}/{REQUIRED_CONSECUTIVE_FRAMES} frames",
                (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 255) if consecutive_count > 0 else (100, 100, 100), 2)

    if fall_detected:
        text = "FALL DETECTED"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 2.0
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        cx = (frame.shape[1] - text_w) // 2
        cy = (frame.shape[0] + text_h) // 2
        cv2.rectangle(frame, (cx - 20, cy - text_h - 20), (cx + text_w + 20, cy + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (cx, cy), font, scale, (0, 0, 255), thickness)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run_fall_detector():
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
    print(f"\nThresholds:")
    print(f"  Body angle  > {ANGLE_THRESHOLD}°")
    print(f"  Hip height  > {HIP_HEIGHT_THRESHOLD}")
    print(f"  Duration    >= {REQUIRED_CONSECUTIVE_FRAMES} consecutive frames\n")
    print("Try bending over, crouching, sitting — watch for false positives.\n")

    previous_hip_y    = None
    consecutive_count = 0
    alert_countdown   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect(mp_image)

        body_angle    = None
        hip_height    = None
        velocity      = None
        fall_detected = alert_countdown > 0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            body_angle = compute_body_angle(landmarks)
            hip_height = compute_hip_height(landmarks)
            velocity   = compute_velocity(hip_height, previous_hip_y)
            previous_hip_y = hip_height

            if check_fall_conditions(body_angle, hip_height):
                consecutive_count += 1
                if consecutive_count >= REQUIRED_CONSECUTIVE_FRAMES:
                    alert_countdown = ALERT_COOLDOWN_FRAMES
                    print(f"FALL DETECTED — angle={body_angle:.1f}° hip_y={hip_height:.3f}")
            else:
                consecutive_count = 0

            # Pass fall_detected to draw_skeleton so bones turn red on alert
            color = (0, 0, 255) if (fall_detected or alert_countdown > 0) else (0, 255, 0)
            pixel_coords = []
            for lm in landmarks:
                px = int(lm.x * frame_width)
                py = int(lm.y * frame_height)
                pixel_coords.append((px, py, lm.visibility))
            for px, py, vis in pixel_coords:
                if vis >= VISIBILITY_THRESHOLD:
                    cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
            for a, b in CONNECTIONS:
                px_a, py_a, vis_a = pixel_coords[a]
                px_b, py_b, vis_b = pixel_coords[b]
                if vis_a >= VISIBILITY_THRESHOLD and vis_b >= VISIBILITY_THRESHOLD:
                    cv2.line(frame, (px_a, py_a), (px_b, py_b), color, 2)

        else:
            consecutive_count = 0
            previous_hip_y = None
            cv2.putText(frame, "No person detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if alert_countdown > 0:
            alert_countdown -= 1
            fall_detected = True

        draw_overlay(frame, body_angle, hip_height, velocity,
                     consecutive_count, fall_detected)

        cv2.imshow("Fall Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_fall_detector()
