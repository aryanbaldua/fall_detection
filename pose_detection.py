import cv2
import time
import matplotlib.pyplot as plt

def display_video():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Print webcam metadata
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Frame shape will be: ({height}, {width}, 3)")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # --- Inspect the first frame ---
        if frame_count == 1:
            print(f"\nFirst frame:")
            print(f"  shape: {frame.shape}")
            print(f"  dtype: {frame.dtype}")
            print(f"  top-left pixel (BGR): {frame[0, 0]}")
            print(f"  center pixel (BGR):   {frame[height//2, width//2]}")

        # pull out inidivudal channels
        if frame_count == 1:
            blue  = frame[:, :, 0]
            green = frame[:, :, 1]
            red   = frame[:, :, 2]
            
            cv2.imshow("Blue channel", blue)
            cv2.imshow("Green channel", green)
            cv2.imshow("Red channel", red)

        if frame_count == 1:
            plt.subplot(1, 2, 1)
            plt.title("Wrong (BGR read as RGB)")
            plt.imshow(frame)

            plt.subplot(1, 2, 2)
            plt.title("Correct (converted to RGB)")
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            plt.show()

        # --- FPS counter ---
        # calculate time difference between new and current frame to get frames per second
        curr_time = time.time()
        fps_actual = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # --- Draw FPS on frame ---
        cv2.putText(
            frame,
            f"FPS: {fps_actual:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # green
            2
        )

        # --- Draw frame number ---
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.imshow("Phase 1", frame)

        # q to quit, space to pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # wait until space is pressed again

    # closes connection to video file and frees memory
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames total.")

display_video()