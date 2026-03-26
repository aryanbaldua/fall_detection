# Anoki Labs — Claude Instructions

## Who I am
I am completely new to computer vision. My goal is to **learn and deeply understand** every concept we work on — not just get working code. I want to understand the "why" behind every decision, not just the "how".

## How to teach me
- **Explain before you code.** Before writing anything non-trivial, briefly explain what approach we're taking and why.
- **Comment the code generously.** Every non-obvious line should have a comment. Assume I don't know what any OpenCV or NumPy function does unless we've already covered it.
- **Use analogies.** I learn better when concepts are connected to things I already understand (general programming, math, everyday experience).
- **Don't skip steps.** If something seems obvious to an expert, it probably isn't obvious to me. Over-explain rather than under-explain.
- **Introduce one concept at a time.** Don't pile on new ideas in a single change. Build up incrementally.

## Code style preferences
- Prefer clear, readable code over clever or compact code.
- Use descriptive variable names (e.g. `confidence_score` not `cs`).
- Add a short comment block at the top of each new script explaining what it does.
- Print useful debug info (shapes, types, values) while learning — we can clean it up later.

## Project context
This project is a progressive, hands-on computer vision curriculum building toward a real-time fall detection system. Each script builds on the last:
1. `pose_detection.py` — webcam capture, frame inspection, BGR channels, FPS overlay
2. `person_detection.py` — YOLOv8 person detection, bounding boxes, confidence scores

## End goal
Build a **real-time fall detection system** — a pipeline that takes webcam/video input, detects a person, estimates their pose, extracts motion features, and triggers an alert when a fall is detected. The final system should work end-to-end in a single script.

## Roadmap (20–27 hours total)

| Phase | Topic | Hours | Status |
|---|---|---|---|
| 1 | Images & Video Basics | 2–3 hrs | In progress |
| 2 | Person Detection (YOLO, bounding boxes) | 2–3 hrs | In progress |
| 3 | Pose Estimation (MediaPipe keypoints) | 3–4 hrs | — |
| 4 | Feature Extraction (body angle, hip height, velocity) | 4–5 hrs | — |
| 5 | Rule-Based Fall Detector (if/else on features) | 2–3 hrs | — |
| 6 | LSTM Classifier (sequence-based ML) | 4–5 hrs | — |
| 7 | Real-Time Pipeline (everything end-to-end) | 3–4 hrs | — |

### Phase 1 — Images & Video Basics
- Image = 3D numpy array (height, width, channels). OpenCV loads as BGR not RGB.
- Color spaces: RGB for display, HSV for color isolation, grayscale to reduce compute.
- Video = sequence of frames via `cap.read()`. FPS matters for computing velocity later.
- **Build:** Display webcam feed with real-time FPS overlay. ✓ (`pose_detection.py`)

### Phase 2 — Person Detection
- Bounding box = 4 numbers: `(x1, y1, x2, y2)` corners around an object.
- YOLO outputs: bounding box + class label + confidence score (0–1) per detection.
- Class index 0 = "person" in the COCO dataset YOLO was trained on.
- Confidence threshold filters weak detections. NMS removes duplicate boxes (YOLO handles this).
- **Build:** Loop through video/webcam, draw bounding box + confidence score per person. ✓ (`person_detection.py`)

### Phase 3 — Pose Estimation
- Keypoints/landmarks = specific named body points (nose, shoulders, elbows, wrists, hips, knees, ankles). MediaPipe gives 33.
- Each landmark has `x, y, z` (normalized 0–1) and a visibility score. Ignore landmarks below ~0.5 visibility.
- Coordinate system: `(0,0)` is top-left, y increases downward. Head has smaller y than hips.
- **Build:** Draw skeleton manually (circles + lines from raw coordinates — no built-in visualizer).

### Phase 4 — Feature Extraction
- Raw coordinates aren't directly useful. Derive meaningful signals:
  - `body_angle` — angle of spine (shoulder midpoint → hip midpoint) relative to vertical. Upright ≈ 0°, fallen ≈ 90°.
  - `hip_height` — y position of hip midpoint. Falls cause rapid increase (y grows downward).
  - `velocity` — change in position between frames: `hip_y[t] - hip_y[t-1]`.
- **Build:** Real-time overlay showing body angle, hip height, and velocity as live numbers on frame.

### Phase 5 — Rule-Based Fall Detector
- Pure if/else logic on extracted features. No ML.
- Simple rule: `if body_angle > 60 and hip_height > 0.7 → FALL`
- Add duration condition: only alert if conditions hold for N consecutive frames.
- **Build:** "FALL DETECTED" red text overlay. Deliberately break it to find failure modes — motivates the LSTM.

### Phase 6 — LSTM Classifier
- Single frame can't distinguish fall from bending over — need sequence context.
- LSTM processes a sliding window of 30 frames, outputs fall probability (0–1).
- Input: `(30, feature_dim)` array. Output: single probability.
- Need labeled data: use UR Fall Detection Dataset (URFD).
- Watch out for class imbalance (falls are rare) — use weighted loss or oversample.
- Evaluate with precision, recall, F1 — not just accuracy.
- **Build:** Trained LSTM saved to disk that takes a `(30, feature_dim)` array and outputs fall probability.

### Phase 7 — Real-Time Pipeline
- Full chain: YOLO → MediaPipe → feature extraction → sliding window → LSTM → alert logic.
- Require 3 consecutive positive predictions before triggering alert (reduces false positives).
- Handle missing detections gracefully (skip frame or hold last known value).
- **Build:** Single script — video input → bounding boxes + skeleton + feature overlays + fall alerts, all end-to-end.

## Key principle
After Phase 5 there is a working product. Phases 6–7 are where the real ML learning happens. The failure modes discovered in Phase 5 are the honest motivation for building the LSTM.

## Reminders
- I am using a Mac with a built-in webcam.
- We use OpenCV (`cv2`) as the primary vision library.
- Always explain what a new library or model does before using it.
