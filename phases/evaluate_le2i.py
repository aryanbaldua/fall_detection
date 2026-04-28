"""
evaluate_le2i.py
----------------
Evaluates the rule-based fall detector against the Le2i dataset using
pre-extracted features from data/le2i_features.csv.

Run extract_features_le2i.py first to generate the CSV.
Then adjust the thresholds below and run this script — it's instant
because MediaPipe has already done the heavy lifting.

Thresholds to tune:
  ANGLE_THRESHOLD             — spine must be tilted more than this (degrees)
  VELOCITY_THRESHOLD          — hip must be dropping faster than this per frame (0–1 scale)
  BBOX_RATIO_THRESHOLD        — skeleton width/height must exceed this (>1 = body more horizontal)
  REQUIRED_CONSECUTIVE_FRAMES — all conditions must hold this many frames in a row
"""

import os
import csv
from collections import defaultdict

# ─────────────────────────────────────────────
# THRESHOLDS — adjust these freely, re-run instantly
# ─────────────────────────────────────────────

ANGLE_THRESHOLD             = 45.0
VELOCITY_THRESHOLD          = 0.03   # hip drops at least 3% of frame height per frame
BBOX_RATIO_THRESHOLD        = 1.0    # skeleton wider than tall
REQUIRED_CONSECUTIVE_FRAMES = 3

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "le2i_features.csv")


# ─────────────────────────────────────────────
# LOAD FEATURES FROM CSV
# ─────────────────────────────────────────────

def load_features():
    """
    Reads le2i_features.csv and groups rows by (scene, video).

    Returns a dict:
      {
        (scene, video): {
          "true_label": "FALL" or "ADL",
          "frames": [(frame_num, angle, hip_height), ...]
        }
      }

    angle and hip_height are floats, or None if MediaPipe had no detection.
    """
    data = {}

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["scene"], row["video"])

            if key not in data:
                data[key] = {
                    "true_label": row["true_label"],
                    "frames": []
                }

            angle      = float(row["angle"])      if row["angle"]      else None
            hip_height = float(row["hip_height"]) if row["hip_height"] else None
            bbox_ratio = float(row["bbox_ratio"]) if row["bbox_ratio"] else None

            data[key]["frames"].append((int(row["frame"]), angle, hip_height, bbox_ratio))

    return data


# ─────────────────────────────────────────────
# APPLY RULE-BASED DETECTOR TO ONE VIDEO
# ─────────────────────────────────────────────

def predict(frames):
    """
    Applies the rule-based fall detection logic to a list of pre-extracted frames.

    Rule: angle > ANGLE_THRESHOLD
          AND hip_velocity > VELOCITY_THRESHOLD  (hip dropping fast)
          AND bbox_ratio   > BBOX_RATIO_THRESHOLD (body wider than tall)
          ... held for REQUIRED_CONSECUTIVE_FRAMES in a row.

    hip_velocity is computed here from consecutive hip_height values —
    it is not stored in the CSV because it is trivially derived.
    Positive velocity means the hips are moving downward (falling).
    """
    consecutive    = 0
    prev_hip_height = None

    for _, angle, hip_height, bbox_ratio in frames:
        # Compute velocity from the change in hip_height between frames
        if hip_height is not None and prev_hip_height is not None:
            hip_velocity = hip_height - prev_hip_height
        else:
            hip_velocity = None

        prev_hip_height = hip_height

        if angle is None or hip_velocity is None or bbox_ratio is None:
            consecutive = 0
            continue

        if (angle       > ANGLE_THRESHOLD    and
            hip_velocity > VELOCITY_THRESHOLD and
            bbox_ratio   > BBOX_RATIO_THRESHOLD):
            consecutive += 1
            if consecutive >= REQUIRED_CONSECUTIVE_FRAMES:
                return True
        else:
            consecutive = 0

    return False


# ─────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────

def run_evaluation():
    if not os.path.exists(CSV_PATH):
        print(f"Features file not found: {CSV_PATH}")
        print("Run extract_features_le2i.py first.")
        return

    print(f"Loading features from {os.path.basename(CSV_PATH)}...")
    data = load_features()
    print(f"Loaded {len(data)} videos.\n")

    print(f"Thresholds: angle > {ANGLE_THRESHOLD}°, "
          f"hip_velocity > {VELOCITY_THRESHOLD}, "
          f"bbox_ratio > {BBOX_RATIO_THRESHOLD}, "
          f"consecutive >= {REQUIRED_CONSECUTIVE_FRAMES} frames")
    print(f"\n{'─'*70}")

    results = []  # (scene, video, true_label, prediction)

    for (scene, video), entry in sorted(data.items()):
        true_label = entry["true_label"]
        prediction = "FALL" if predict(entry["frames"]) else "ADL"
        correct    = prediction == true_label
        status     = "✓" if correct else "✗"

        print(f"  {status}  [{scene}]  {video:<20}  truth={true_label:<5}  pred={prediction}")
        results.append((scene, video, true_label, prediction))

    # ─────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────

    TP = sum(1 for _, _, t, p in results if p == "FALL" and t == "FALL")
    FP = sum(1 for _, _, t, p in results if p == "FALL" and t == "ADL")
    TN = sum(1 for _, _, t, p in results if p == "ADL"  and t == "ADL")
    FN = sum(1 for _, _, t, p in results if p == "ADL"  and t == "FALL")

    total     = len(results)
    accuracy  = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP)    if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN)    if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'─'*70}")
    print(f"RESULTS — {total} videos")
    print(f"{'─'*70}")
    print(f"  True Positives  (fall caught):      {TP}")
    print(f"  False Negatives (fall missed):      {FN}  ← missed falls")
    print(f"  True Negatives  (ADL correct):      {TN}")
    print(f"  False Positives (false alarms):     {FP}")
    print(f"{'─'*70}")
    print(f"  Accuracy:   {accuracy:.1%}")
    print(f"  Precision:  {precision:.1%}   (when we say FALL, how often right?)")
    print(f"  Recall:     {recall:.1%}   (of all falls, how many caught?)")
    print(f"  F1 Score:   {f1:.1%}   (balance of precision + recall)")
    print(f"{'─'*70}")

    # Per-scene breakdown
    print(f"\nPer-scene breakdown:")
    for scene_name in ["Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"]:
        scene_results = [(t, p) for s, _, t, p in results if s == scene_name]
        if not scene_results:
            continue
        s_tp  = sum(1 for t, p in scene_results if p == "FALL" and t == "FALL")
        s_fp  = sum(1 for t, p in scene_results if p == "FALL" and t == "ADL")
        s_tn  = sum(1 for t, p in scene_results if p == "ADL"  and t == "ADL")
        s_fn  = sum(1 for t, p in scene_results if p == "ADL"  and t == "FALL")
        s_acc = (s_tp + s_tn) / len(scene_results)
        print(f"  {scene_name:<16}  {len(scene_results):>3} videos  "
              f"TP={s_tp} FP={s_fp} TN={s_tn} FN={s_fn}  acc={s_acc:.0%}")

    if FN > 0:
        print(f"\nMissed falls:")
        for s, v, t, p in results:
            if t == "FALL" and p == "ADL":
                print(f"  [{s}] {v}")

    if FP > 0:
        print(f"\nFalse alarms:")
        for s, v, t, p in results:
            if t == "ADL" and p == "FALL":
                print(f"  [{s}] {v}")


if __name__ == "__main__":
    run_evaluation()
