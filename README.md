# Fall Detection

A hands-on computer vision learning project building toward a real-time fall detection system.

This repo documents my learning journey from zero computer vision knowledge to a working fall detector. Each phase builds on the last.

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

## Setup

```bash
pip install opencv-python ultralytics mediapipe
```

Download model files (not included in repo):
```bash
# YOLOv8
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# MediaPipe Pose Landmarker
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o pose_landmarker_full.task
```
