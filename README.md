# Fall Detection

A computer vision project that detects when a person falls in a video. The pipeline runs MediaPipe pose estimation on each frame, derives motion features (body angle, hip height, hip velocity), and classifies the result.

## Approaches

| Approach | Status | Notebook | Idea |
|---|---|---|---|
| Rule-based | Done | [`notebooks/rule_based.ipynb`](notebooks/rule_based.ipynb) | Trigger an alert when body angle and hip height cross hand-tuned thresholds for several consecutive frames |
| LSTM | Coming | `notebooks/lstm.ipynb` | Train a sequence model on a sliding window of features so the classifier can learn what a fall *looks like* over time |

Both approaches share the same upstream feature extraction. The rule-based notebook is evaluated on Le2i; an additional held-out evaluation on URFD lives in [`notebooks/evaluation_set.ipynb`](notebooks/evaluation_set.ipynb).

## Setup

```bash
pip install opencv-python ultralytics mediapipe
```

Model weights are gitignored — download them once:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
curl -L "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -o models/pose_landmarker_full.task
```

## Datasets

Raw videos are gitignored; only the extracted feature CSVs are committed.

- **Le2i** — primary training/eval set. Place raw videos under `data/Le2i/`.
- **URFD** — held-out evaluation set. [Project page](https://fenix.ur.edu.pl/mkepski/ds/uf.html). Download with:
  ```bash
  bash scripts/download_urfd.sh
  ```
