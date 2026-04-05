"""
threshold_sweep.py
------------------
Runs the rule-based fall detector across every combination of threshold values
and prints a table showing how precision, recall, and F1 change.

IMPORTANT — train/test split:
  We split the dataset 80/20 BEFORE the sweep.
  Thresholds are picked using only the train set.
  The best thresholds are then evaluated ONCE on the held-out test set.

  This prevents data leakage — the test set numbers are honest because
  the thresholds were never tuned against them.

The three parameters we sweep:
  - ANGLE_THRESHOLD            (how tilted the spine must be)
  - HIP_HEIGHT_THRESHOLD       (how low the hips must be)
  - REQUIRED_CONSECUTIVE_FRAMES (how many frames in a row conditions must hold)
"""

import os
import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from itertools import product

from feature_extraction import (
    compute_body_angle,
    compute_hip_height,
    MODEL_PATH,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "URFD")

# ─────────────────────────────────────────────
# THRESHOLD RANGES TO SWEEP
# ─────────────────────────────────────────────
ANGLE_THRESHOLDS      = [30, 45, 60, 75]
HIP_HEIGHT_THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.80]
CONSECUTIVE_FRAMES    = [5, 10, 15, 20]

# Fraction of data held out for final evaluation
TEST_SPLIT = 0.20

# Random seed makes the split reproducible — same split every run
RANDOM_SEED = 42


def load_sequences():
    """
    Loads all sequence folders and their ground truth labels.
    Returns a list of (folder_path, label) where label is 1=fall, 0=no fall.
    """
    sequences = []
    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("fall-"):
            sequences.append((path, 1))
        elif name.startswith("adl-"):
            sequences.append((path, 0))
    return sequences


def train_test_split(sequences, test_fraction, seed):
    """
    Splits sequences into train and test sets.

    We split fall and ADL sequences SEPARATELY (this is called stratified
    splitting) to ensure both sets have the same fall/ADL ratio.

    Why stratified? If we split randomly across all sequences, we might
    get unlucky and put all the falls in train and none in test (or vice
    versa). Stratifying guarantees the ratio is preserved.

    Example with 21 falls and 32 ADLs at 80/20:
      Falls: 17 train, 4 test
      ADLs:  26 train, 6 test
    """
    rng = random.Random(seed)

    falls = [(p, l) for p, l in sequences if l == 1]
    adls  = [(p, l) for p, l in sequences if l == 0]

    # Shuffle each group independently before splitting
    rng.shuffle(falls)
    rng.shuffle(adls)

    def split(items, fraction):
        n_test = max(1, round(len(items) * fraction))
        return items[n_test:], items[:n_test]  # (train, test)

    fall_train, fall_test = split(falls, test_fraction)
    adl_train,  adl_test  = split(adls,  test_fraction)

    train = fall_train + adl_train
    test  = fall_test  + adl_test

    return train, test


def extract_features_for_sequence(sequence_path, landmarker):
    """
    Runs MediaPipe on every frame and returns a list of (body_angle, hip_height).
    We pre-extract once and reuse across all 80 threshold combinations.
    """
    frames = sorted([f for f in os.listdir(sequence_path) if f.endswith(".png")])
    frame_features = []

    for filename in frames:
        frame_path = os.path.join(sequence_path, filename)
        bgr = cv2.imread(frame_path)
        if bgr is None:
            frame_features.append((None, None))
            continue

        rgb      = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            frame_features.append((None, None))
            continue

        landmarks  = result.pose_landmarks[0]
        body_angle = compute_body_angle(landmarks)
        hip_height = compute_hip_height(landmarks)
        frame_features.append((body_angle, hip_height))

    return frame_features


def classify_sequence(frame_features, angle_thresh, hip_thresh, consec_thresh):
    """Applies rule-based logic to pre-extracted features."""
    consecutive_count = 0
    for body_angle, hip_height in frame_features:
        if body_angle is None or hip_height is None:
            consecutive_count = 0
            continue
        if body_angle > angle_thresh and hip_height > hip_thresh:
            consecutive_count += 1
            if consecutive_count >= consec_thresh:
                return True
        else:
            consecutive_count = 0
    return False


def compute_metrics(predictions, ground_truths):
    """Computes precision, recall, F1, and accuracy."""
    TP = sum(1 for p, t in zip(predictions, ground_truths) if p == 1 and t == 1)
    FP = sum(1 for p, t in zip(predictions, ground_truths) if p == 1 and t == 0)
    TN = sum(1 for p, t in zip(predictions, ground_truths) if p == 0 and t == 0)
    FN = sum(1 for p, t in zip(predictions, ground_truths) if p == 0 and t == 1)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / len(predictions) if predictions else 0.0

    return precision, recall, f1, accuracy, TP, FP, TN, FN


def run_sweep():
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

    # --- Load and split sequences ---
    all_sequences = load_sequences()
    train_seqs, test_seqs = train_test_split(all_sequences, TEST_SPLIT, RANDOM_SEED)

    print(f"Total sequences: {len(all_sequences)}")
    print(f"Train: {len(train_seqs)} ({sum(l for _, l in train_seqs)} falls, {sum(1-l for _, l in train_seqs)} ADLs)")
    print(f"Test:  {len(test_seqs)} ({sum(l for _, l in test_seqs)} falls, {sum(1-l for _, l in test_seqs)} ADLs)")
    print(f"\nTest sequences (locked away — not used for threshold selection):")
    for path, label in sorted(test_seqs, key=lambda x: os.path.basename(x[0])):
        print(f"  {'FALL' if label else 'ADL ':4}  {os.path.basename(path)}")

    # --- Pre-extract features for TRAIN set only ---
    print(f"\nExtracting features from {len(train_seqs)} train sequences...")
    train_features    = []
    train_labels      = []

    for i, (path, label) in enumerate(train_seqs):
        name = os.path.basename(path)
        print(f"  [{i+1}/{len(train_seqs)}] {name}")
        train_features.append(extract_features_for_sequence(path, landmarker))
        train_labels.append(label)

    # --- Sweep thresholds on TRAIN set ---
    n_combos = len(ANGLE_THRESHOLDS) * len(HIP_HEIGHT_THRESHOLDS) * len(CONSECUTIVE_FRAMES)
    print(f"\nRunning {n_combos} threshold combinations on train set...\n")

    train_results = []

    for angle_thresh, hip_thresh, consec_thresh in product(
        ANGLE_THRESHOLDS, HIP_HEIGHT_THRESHOLDS, CONSECUTIVE_FRAMES
    ):
        predictions = [
            1 if classify_sequence(f, angle_thresh, hip_thresh, consec_thresh) else 0
            for f in train_features
        ]
        precision, recall, f1, accuracy, TP, FP, TN, FN = compute_metrics(predictions, train_labels)
        train_results.append({
            "angle": angle_thresh, "hip": hip_thresh, "consec": consec_thresh,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        })

    train_results.sort(key=lambda r: r["f1"], reverse=True)

    # --- Print train results ---
    print("TRAIN SET RESULTS (sorted by F1):")
    header = f"{'Angle':>6}  {'Hip':>5}  {'Consec':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  TP  FP  TN  FN"
    print(header)
    print("─" * len(header))
    for r in train_results[:15]:  # show top 15
        print(
            f"{r['angle']:>6}°  {r['hip']:>5.2f}  {r['consec']:>6}  "
            f"{r['precision']:>6.1%}  {r['recall']:>6.1%}  {r['f1']:>6.1%}  "
            f"{r['TP']:>2}  {r['FP']:>2}  {r['TN']:>2}  {r['FN']:>2}"
        )

    # --- Pick best thresholds from train set ---
    best = train_results[0]
    print(f"\nBest thresholds (by F1 on train set): angle={best['angle']}°  hip={best['hip']}  consec={best['consec']}")
    print(f"Train performance: precision={best['precision']:.1%}  recall={best['recall']:.1%}  F1={best['f1']:.1%}")

    # ─────────────────────────────────────────────
    # FINAL EVALUATION ON TEST SET
    # This is the honest number. Run ONCE with the best thresholds.
    # ─────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'─'*50}")

    print(f"Extracting features from {len(test_seqs)} test sequences...")
    test_features = []
    test_labels   = []
    for path, label in test_seqs:
        test_features.append(extract_features_for_sequence(path, landmarker))
        test_labels.append(label)

    test_preds = [
        1 if classify_sequence(f, best['angle'], best['hip'], best['consec']) else 0
        for f in test_features
    ]
    precision, recall, f1, accuracy, TP, FP, TN, FN = compute_metrics(test_preds, test_labels)

    print(f"\nThresholds used: angle={best['angle']}°  hip={best['hip']}  consec={best['consec']}")
    print(f"\nPer-sequence results:")
    for (path, label), pred in zip(test_seqs, test_preds):
        name   = os.path.basename(path)
        truth  = "FALL" if label else "NO FALL"
        result = "FALL" if pred  else "NO FALL"
        status = "✓" if label == pred else "✗"
        print(f"  {status}  {name:<30}  truth={truth:<8}  pred={result}")

    print(f"\n  True Positives  (fall caught):    {TP}")
    print(f"  False Negatives (fall missed):    {FN}")
    print(f"  True Negatives  (ADL correct):    {TN}")
    print(f"  False Positives (ADL wrong):      {FP}")
    print(f"{'─'*50}")
    print(f"  Accuracy:   {accuracy:.1%}")
    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1 Score:   {f1:.1%}")
    print(f"{'─'*50}")
    print(f"\nThese are the honest numbers — test set was never used for threshold selection.")


if __name__ == "__main__":
    run_sweep()
