"""
download_urfd.py
----------------
Downloads the UR Fall Detection Dataset (URFD) videos from
https://fenix.ur.edu.pl/mkepski/ds/uf.html into data/URFD/Videos/.

Why we only grab cam0:
  URFD records each fall from two angles —
    cam0 = floor-mounted, parallel to the floor (CCTV-style)
    cam1 = ceiling-mounted, looking straight down
  Our rule-based detector relies on body_angle (spine vs. vertical) and
  hip_height (y position in frame). Both features are meaningless from a
  top-down view — a person lying on the floor looks just like a person
  standing when seen from directly above. cam0 matches Le2i's perspective,
  so the rules tuned on Le2i can transfer here. cam1 cannot be used.

What we download:
  - 30 fall videos: fall-01-cam0.mp4 ... fall-30-cam0.mp4
  - 40 ADL videos:  adl-01-cam0.mp4  ... adl-40-cam0.mp4
  Total: 70 files, roughly 1 MB each (~80 MB).

The script is idempotent — already-downloaded files are skipped.
"""

import os
import sys
import subprocess

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
# Resolve everything relative to this file so it works regardless of CWD.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "..", "data", "URFD", "Videos")

# Base URL where every file lives. Filenames are appended directly.
BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data/"

# URFD has 30 fall sequences and 40 ADL sequences (numbered starting from 1).
NUM_FALLS = 30
NUM_ADLS  = 40


# ─────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────

def download_one(filename, dest_dir):
    """
    Downloads a single file from BASE_URL/filename into dest_dir/filename.

    If the file already exists locally, it is skipped (idempotent).
    Returns one of: "downloaded", "skipped", "failed".
    """
    url       = BASE_URL + filename
    dest_path = os.path.join(dest_dir, filename)

    # Skip if we've already pulled this file in a previous run.
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return "skipped"

    # We shell out to curl rather than using urllib because Python's default
    # SSL context on macOS often doesn't trust this server's cert chain.
    # curl uses the system keychain and just works.
    #   -s : silent (no progress bar — we print our own)
    #   -S : but still show errors
    #   -L : follow redirects
    #   -f : fail with non-zero exit on HTTP errors (like 404)
    #   -o : write to this output path
    result = subprocess.run(
        ["curl", "-sSLf", "-o", dest_path, url],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return "downloaded"
    # Clean up any partial/empty file curl may have left behind on failure.
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == 0:
        os.remove(dest_path)
    print(f"  [ERROR] {filename}: {result.stderr.strip()}")
    return "failed"


def build_file_list():
    """
    Returns a list of every cam0 mp4 we want to grab.

    URFD numbers files with zero-padded two-digit indices:
      fall-01-cam0.mp4, fall-02-cam0.mp4, ..., fall-30-cam0.mp4
      adl-01-cam0.mp4,  adl-02-cam0.mp4,  ..., adl-40-cam0.mp4
    """
    files = []
    for i in range(1, NUM_FALLS + 1):
        files.append(f"fall-{i:02d}-cam0.mp4")   # :02d pads single digits with 0
    for i in range(1, NUM_ADLS + 1):
        files.append(f"adl-{i:02d}-cam0.mp4")
    return files


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    # Make the destination directory; exist_ok=True means re-runs are fine.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving videos to: {OUTPUT_DIR}")

    files = build_file_list()
    print(f"Total files to fetch: {len(files)}\n")

    # Tally results so we can print a summary at the end.
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}

    for idx, fname in enumerate(files, start=1):
        # Print progress before the download so the user sees activity.
        print(f"[{idx:2d}/{len(files)}] {fname} ... ", end="", flush=True)
        result = download_one(fname, OUTPUT_DIR)
        print(result)
        counts[result] += 1

    print("\n--- Summary ---")
    print(f"  Downloaded: {counts['downloaded']}")
    print(f"  Skipped (already present): {counts['skipped']}")
    print(f"  Failed:     {counts['failed']}")

    # Non-zero exit code if anything failed — useful when running from a script.
    if counts["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    run()
