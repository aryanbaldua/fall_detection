"""
evaluate_rule_based.py
----------------------
Runs the rule-based fall detector against every sequence in the URFD dataset
and reports how well it performs.

For each sequence:
  - Loads frames in order (PNG images)
  - Runs MediaPipe pose estimation on each frame
  - Applies the same rule-based logic as rule_based_detector.py
  - Records the prediction: FALL or NO FALL

Ground truth comes from the folder name:
  - fall-XX-cam0-rgb → label: FALL
  - adl-XX-cam0-rgb  → label: NO FALL

Then computes:
  - Precision  — of all sequences we called FALL, what fraction actually were?
  - Recall     — of all actual falls, what fraction did we catch?
  - F1         — harmonic mean of precision and recall
  - A per-sequence breakdown of correct/wrong predictions
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import feature computation functions from Phase 4
from feature_extraction import (
    compute_body_angle,
    compute_hip_height,
    compute_velocity,
    MODEL_PATH,
)

# ─────────────────────────────────────────────
# SAME THRESHOLDS AS rule_based_detector.py
# If you change them there, change them here too.
# ─────────────────────────────────────────────
ANGLE_THRESHOLD            = 45.0
HIP_HEIGHT_THRESHOLD       = 0.65
REQUIRED_CONSECUTIVE_FRAMES = 10

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "URFD")


def evaluate_sequence(sequence_dir, landmarker):
    """
    Runs the rule-based detector on a single sequence folder.

    Returns True if a fall was detected at any point in the sequence,
    False otherwise.

    This mirrors exactly what rule_based_detector.py does frame by frame —
    the only difference is we're reading PNG files instead of a webcam.
    """
    # Get all PNG frames in this folder, sorted numerically by filename
    frames = sorted([
        f for f in os.listdir(sequence_dir) if f.endswith(".png")
    ])

    if not frames:
        return False

    consecutive_count = 0  # how many frames in a row conditions have been met
    previous_hip_y    = None

    for filename in frames:
        frame_path = os.path.join(sequence_dir, filename)

        # Read the image with OpenCV — returns a BGR numpy array
        bgr_frame = cv2.imread(frame_path)
        if bgr_frame is None:
            continue  # skip corrupted or missing frames

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            # No person detected in this frame — reset streak
            consecutive_count = 0
            previous_hip_y    = None
            continue

        landmarks  = result.pose_landmarks[0]
        body_angle = compute_body_angle(landmarks)
        hip_height = compute_hip_height(landmarks)

        if body_angle is None or hip_height is None:
            consecutive_count = 0
            previous_hip_y    = None
            continue

        previous_hip_y = hip_height

        # Check both conditions simultaneously
        angle_triggered      = body_angle  > ANGLE_THRESHOLD
        hip_height_triggered = hip_height  > HIP_HEIGHT_THRESHOLD

        if angle_triggered and hip_height_triggered:
            consecutive_count += 1
            if consecutive_count >= REQUIRED_CONSECUTIVE_FRAMES:
                return True  # fall detected — no need to process remaining frames
        else:
            consecutive_count = 0  # reset — conditions must be consecutive

    return False  # made it through all frames without triggering


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}.")
        return

    # --- Load MediaPipe ---
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # --- Find all sequence folders ---
    all_sequences = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and (d.startswith("fall-") or d.startswith("adl-"))
    ])

    print(f"Found {len(all_sequences)} sequences")
    print(f"Thresholds: angle > {ANGLE_THRESHOLD}°, hip_y > {HIP_HEIGHT_THRESHOLD}, "
          f"duration >= {REQUIRED_CONSECUTIVE_FRAMES} frames\n")

    # ─────────────────────────────────────────────
    # Run detector on every sequence
    # ─────────────────────────────────────────────
    results = []  # list of (sequence_name, ground_truth, prediction)

    for sequence_name in all_sequences:
        sequence_dir  = os.path.join(DATA_DIR, sequence_name)
        ground_truth  = "FALL" if sequence_name.startswith("fall-") else "NO FALL"
        prediction    = "FALL" if evaluate_sequence(sequence_dir, landmarker) else "NO FALL"
        correct       = prediction == ground_truth

        status = "✓" if correct else "✗"
        print(f"  {status}  {sequence_name:<30}  truth={ground_truth:<8}  pred={prediction}")

        results.append((sequence_name, ground_truth, prediction))

    # ─────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────
    # Confusion matrix counts:
    #   TP = predicted FALL,    actually FALL    ← correct detection
    #   FP = predicted FALL,    actually NO FALL ← false alarm
    #   TN = predicted NO FALL, actually NO FALL ← correct dismissal
    #   FN = predicted NO FALL, actually FALL    ← missed fall
    TP = sum(1 for _, truth, pred in results if pred == "FALL"    and truth == "FALL")
    FP = sum(1 for _, truth, pred in results if pred == "FALL"    and truth == "NO FALL")
    TN = sum(1 for _, truth, pred in results if pred == "NO FALL" and truth == "NO FALL")
    FN = sum(1 for _, truth, pred in results if pred == "NO FALL" and truth == "FALL")

    total = len(results)
    accuracy  = (TP + TN) / total if total > 0 else 0

    # Precision: when we say FALL, how often are we right?
    # If we never predict FALL, precision is undefined — we use 0.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall: of all actual falls, how many did we catch?
    # This is the most important metric for a safety system.
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1: balances precision and recall into a single number
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'─'*50}")
    print(f"RESULTS ({total} sequences)")
    print(f"{'─'*50}")
    print(f"  True Positives  (fall caught):    {TP}")
    print(f"  False Negatives (fall missed):    {FN}")
    print(f"  True Negatives  (ADL correct):    {TN}")
    print(f"  False Positives (ADL wrong):      {FP}")
    print(f"{'─'*50}")
    print(f"  Accuracy:   {accuracy:.1%}")
    print(f"  Precision:  {precision:.1%}   (when we say FALL, how often right?)")
    print(f"  Recall:     {recall:.1%}   (of all falls, how many caught?)")
    print(f"  F1 Score:   {f1:.1%}   (balance of precision + recall)")
    print(f"{'─'*50}")

    if FN > 0:
        print(f"\nMissed falls (false negatives):")
        for name, truth, pred in results:
            if truth == "FALL" and pred == "NO FALL":
                print(f"  {name}")

    if FP > 0:
        print(f"\nFalse alarms (false positives):")
        for name, truth, pred in results:
            if truth == "NO FALL" and pred == "FALL":
                print(f"  {name}")


if __name__ == "__main__":
    run_evaluation()
