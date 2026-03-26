"""
person_detection.py
-------------------
Uses YOLOv8 (You Only Look Once) to detect people in real-time from the webcam.

For every detected person it draws:
  - A green bounding box
  - A confidence score above the box

On first run, ultralytics will automatically download the YOLOv8 model weights
(~6MB for the nano model). After that it runs from cache.
"""

from ultralytics import YOLO
import cv2

# --- Load the model ---
# YOLOv8 comes in several sizes. We use 'yolov8n' — the nano (smallest) version.
# Smaller = faster but slightly less accurate. Good enough for learning.
#
# Size options (smallest to largest / fastest to most accurate):
#   yolov8n  (nano)
#   yolov8s  (small)
#   yolov8m  (medium)
#   yolov8l  (large)
#   yolov8x  (extra large)
model = YOLO("yolov8n.pt")

# YOLO can detect 80 different object types (people, cars, dogs, chairs, etc.)
# Each class has an index. Class 0 is "person".
PERSON_CLASS_ID = 0

def run_person_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # --- Run YOLO on the frame ---
        # YOLO looks at the entire image in one forward pass through the neural network.
        # 'results' is a list — one entry per image. Since we pass one frame, results[0].
        #
        # verbose=False suppresses the per-frame console output YOLO prints by default.
        results = model(frame, verbose=False)
        detections = results[0]  # get results for our single frame

        person_count = 0

        # --- Loop through every detected object ---
        # detections.boxes contains all detected objects in this frame.
        # Each box has: xyxy (coordinates), conf (confidence), cls (class id)
        for box in detections.boxes:

            class_id = int(box.cls)          # which object type was detected
            confidence = float(box.conf)     # how confident YOLO is (0.0 to 1.0)

            # Skip anything that isn't a person
            if class_id != PERSON_CLASS_ID:
                continue

            person_count += 1

            # box.xyxy gives [x1, y1, x2, y2] — top-left and bottom-right corners
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- Draw the bounding box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # --- Draw the confidence label ---
            label = f"Person: {confidence:.2f}"

            # Measure text size so we can draw a background behind it
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Filled black rectangle as background for the text
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 0, 0), -1)

            # White text on top of the black background
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Show person count in top-left corner ---
        cv2.putText(
            frame,
            f"People detected: {person_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 200, 255),
            2
        )

        cv2.imshow("Person Detection (YOLOv8)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

run_person_detection()
