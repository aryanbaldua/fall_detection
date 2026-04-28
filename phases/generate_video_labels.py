"""
generate_video_labels.py
------------------------
Reads the Le2i annotation files and writes one row per video:
    scene, video, true_label   (1 = FALL, 0 = ADL)

Output: data/le2i_video_labels.csv
"""

import os
import csv
import glob

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Le2i")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "le2i_video_labels.csv")

SCENES = [
    ("Coffee_room_01", "Coffee_room_01/Coffee_room_01/Videos", "Coffee_room_01/Coffee_room_01/Annotation_files"),
    ("Coffee_room_02", "Coffee_room_02/Coffee_room_02/Videos", "Coffee_room_02/Coffee_room_02/Annotations_files"),
    ("Home_01",        "Home_01/Home_01/Videos",               "Home_01/Home_01/Annotation_files"),
    ("Home_02",        "Home_02/Home_02/Videos",               "Home_02/Home_02/Annotation_files"),
]

# Returns 1 (FALL) or 0 (ADL) by reading the annotation file — same logic as extract_features_le2i.py
def get_label(ann_path):
    with open(ann_path) as f:
        lines = f.readlines()

    first = lines[0].strip()

    if "," not in first:
        fall_start = int(first)
        fall_end   = int(lines[1].strip())
        return 1 if (fall_start != 0 and fall_end != 0) else 0

    # No integer header — known edge cases in Le2i that are always falls
    return 1


def run():
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["scene", "video", "true_label"])

        for scene_name, vid_rel, ann_rel in SCENES:
            vid_dir = os.path.join(DATA_DIR, vid_rel)
            ann_dir = os.path.join(DATA_DIR, ann_rel)

            for vpath in sorted(glob.glob(os.path.join(vid_dir, "*.avi"))):
                stem     = os.path.basename(vpath).replace(".avi", "")
                ann_path = os.path.join(ann_dir, stem + ".txt")

                if not os.path.exists(ann_path):
                    print(f"  [WARNING] No annotation for {stem} in {scene_name} — skipping")
                    continue

                label = get_label(ann_path)
                writer.writerow([scene_name, stem, label])
                print(f"  {scene_name}, {stem} -> {label}")

    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
