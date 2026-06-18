# Data Directory

This directory is reserved for local dataset files used during development.

## Structure

```
data/
├── raw/       # Place competition video files here (*.mp4, *.avi, *.mov)
├── frames/    # Optional local extracted frames
└── labels/    # Optional local annotation files
```

## Notes

- All subdirectories are **gitignored** to avoid committing large files.
- Video files must follow the naming convention used in the competition dataset.
- Keep generated outputs under `outputs/`; keep committed test fixtures under `tests/fixtures/`.

## Validation

```bash
python scripts/validate_results_json.py tests/fixtures/dummy_results.json
```
