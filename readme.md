# Australian Dashcam Plate Recognition

This is a simple CLI app that scans dashcam footage and outputs a list of
Australian registration plates detected in the video.

It uses OpenCV to sample frames and find plate-like regions, then runs OCR on
those crops. Results are filtered with Australian plate heuristics and
deduplicated across frames.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python app.py path/to/dashcam.mp4 --output plates.json
```

## GUI

Launch the desktop app:

```bash
python gui.py
```

If Tkinter is missing (common on minimal Linux installs), install it with your
system package manager (for example `sudo apt-get install python3-tk`).

## Usage

```bash
python app.py path/to/dashcam.mp4 \
  --sample-fps 2 \
  --min-confidence 0.45 \
  --max-candidates 10 \
  --output plates.json
```

### Useful options

- `--sample-fps` controls how many frames per second are sampled.
- `--min-confidence` raises or lowers OCR confidence filtering.
- `--strict-patterns` only accepts known AU plate patterns.
- `--debug-dir` writes plate crops for tuning OCR.

## Output format

JSON output includes a list of plates with counts and timestamps:

```json
{
  "video": "dashcam.mp4",
  "frames_processed": 1120,
  "sample_fps": 2.0,
  "processing_seconds": 41.2,
  "plates_found": 4,
  "plates": [
    {
      "plate": "ABC123",
      "count": 6,
      "first_seen": 4.32,
      "last_seen": 15.67,
      "avg_confidence": 0.71,
      "sample_timestamps": [4.32, 5.31, 8.02]
    }
  ]
}
```

CSV output is also supported when the output file ends with `.csv`.

## Notes for Australian plates

The filter is tuned for common Australian formats and requires 4-7 characters
with at least one letter and one digit. You can tighten this by enabling
`--strict-patterns` or editing `AU_PLATE_PATTERNS` in `app.py`.

## Performance tips

- Lower `--sample-fps` to reduce processing time.
- Use `--debug-dir` to capture crops and tune the heuristics.
- If you have a supported GPU, add `--gpu` for faster OCR.