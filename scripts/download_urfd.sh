#!/bin/bash
# Downloads all camera 0 RGB sequences from the UR Fall Detection Dataset
# Falls: 30 sequences, ADLs: 40 sequences

BASE_URL="https://fenix.ur.edu.pl/mkepski/ds/data"
DEST="/Users/aryanbaldua/Anoki_Labs/data/URFD"

echo "Downloading fall sequences (1–30)..."
for i in $(seq -w 1 30); do
    FILE="fall-${i}-cam0-rgb.zip"
    echo "  Downloading $FILE..."
    curl -s -o "$DEST/$FILE" "$BASE_URL/$FILE"
    unzip -q "$DEST/$FILE" -d "$DEST"
    rm "$DEST/$FILE"
done

echo "Downloading ADL sequences (1–40)..."
for i in $(seq -w 1 40); do
    FILE="adl-${i}-cam0-rgb.zip"
    echo "  Downloading $FILE..."
    curl -s -o "$DEST/$FILE" "$BASE_URL/$FILE"
    unzip -q "$DEST/$FILE" -d "$DEST"
    rm "$DEST/$FILE"
done

echo "Done. Contents of $DEST:"
ls "$DEST"
