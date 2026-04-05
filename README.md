# Fall Detection

A hands-on computer vision learning project building toward a real-time fall detection system. Starting from zero knowledge of computer vision, each phase introduces one new concept and builds directly on the last.

The goal is a pipeline that takes webcam or video input, detects a person, estimates their pose, extracts motion features, and triggers an alert when a fall is detected.

## Roadmap

| Phase | Topic | File | Description |
|---|---|---|---|
| 1 | Images & Video Basics | `pose_detection.py` | Learn how images and video work as numpy arrays in OpenCV |
| 2 | Person Detection | `person_detection.py` | Use YOLOv8 to draw bounding boxes around people with confidence scores |
| 3 | Pose Estimation | `pose_estimation.py` | Use MediaPipe to detect 33 body keypoints and draw a skeleton |
| 4 | Feature Extraction | `feature_extraction.py` | Compute body angle, hip height, and velocity from raw keypoints |
| 5 | Rule-Based Detector | `rule_based_detector.py` | Detect falls using simple if/else logic on extracted features |
| 6 | LSTM Classifier | `lstm_classifier.py` | Train a neural network to classify falls from sequences of frames |
| 7 | Real-Time Pipeline | `pipeline.py` | Connect everything into a single end-to-end live detection system |

## Evaluation

Before building the LSTM, the rule-based detector was evaluated against the [UR Fall Detection Dataset (URFD)](https://fenix.ur.edu.pl/mkepski/ds/uf.html) — a standard benchmark containing 30 fall sequences and 40 activities of daily living (ADL) such as walking, sitting, and bending.

A threshold grid search was run across 80 combinations of:
- Body angle threshold (30°, 45°, 60°, 75°)
- Hip height threshold (0.50, 0.60, 0.65, 0.70, 0.80)
- Required consecutive frames (5, 10, 15, 20)

Using a stratified 80/20 train/test split to avoid data leakage, the best thresholds (`angle > 60°`, `hip > 0.60`, `consec >= 5`) achieved:

| Metric | Score |
|---|---|
| Precision | 100.0% |
| Recall | 75.0% |
| F1 | 85.7% |

The recall ceiling of ~75–85% across all threshold combinations motivates the LSTM in Phase 6 — some falls simply don't produce the expected body angle and hip height signature, and no threshold tuning can fix that.

## Dataset

Download the URFD camera 0 RGB sequences and place them in `data/URFD/`:

```bash
bash scripts/download_urfd.sh
```

Or download manually from [fenix.ur.edu.pl/mkepski/ds/uf.html](https://fenix.ur.edu.pl/mkepski/ds/uf.html).

## Setup

```bash
pip install opencv-python ultralytics mediapipe
```

Download model files (not included in repo):
```bash
# YOLOv8
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# MediaPipe Pose Landmarker
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o models/pose_landmarker_full.task
```
