# Data Directory

This directory holds all dataset files used by the pipeline.

## Structure

```
data/
├── raw/       ← Place competition video files here (*.mp4, *.avi, *.mov)
├── frames/    ← Extracted frames will be saved here automatically
└── labels/    ← Ground truth annotation files (JSON / YOLO .txt format)
```

## Notes

- All subdirectories are **gitignored** to avoid committing large files.
- Video files must follow the naming convention used in the competition dataset.
- Label files should be in COCO JSON or YOLO annotation format.

## Quick start

```bash
# Copy your video files into data/raw/
cp /path/to/competition/video.mp4 data/raw/

# Extract frames (1 frame per second by default)
python scripts/extract_frames.py --video data/raw/video.mp4 --output data/frames/video
```
