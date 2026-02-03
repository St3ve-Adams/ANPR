#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

try:
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime
    easyocr = None


AU_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{3}\d{3}$"),
    re.compile(r"^[A-Z]{2}\d{3}$"),
    re.compile(r"^[A-Z]{1,3}\d{2,4}$"),
    re.compile(r"^\d{1,2}[A-Z]{2,3}\d{1,2}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}$"),
]

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def build_reader(use_gpu: bool) -> "easyocr.Reader":
    if easyocr is None:
        raise SystemExit(
            "easyocr is not installed. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        )
    return easyocr.Reader(["en"], gpu=use_gpu)


def normalize_plate(text: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return cleaned


def is_valid_au_plate(text: str, strict_patterns: bool) -> bool:
    if not (4 <= len(text) <= 7):
        return False
    if not text.isalnum():
        return False
    if not any(char.isalpha() for char in text):
        return False
    if not any(char.isdigit() for char in text):
        return False
    if any(pattern.match(text) for pattern in AU_PLATE_PATTERNS):
        return True
    return not strict_patterns


def iter_sampled_frames(
    video_path: str,
    sample_fps: float,
    max_frames: Optional[int],
) -> Iterable[Tuple[int, float, "cv2.Mat"]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = 1
    if sample_fps > 0 and sample_fps < fps:
        frame_interval = max(1, int(round(fps / sample_fps)))

    frame_idx = 0
    yielded = 0
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        timestamp = frame_idx / fps if fps else 0.0
        yield frame_idx, timestamp, frame
        frame_idx += 1
        yielded += 1
        if max_frames and yielded >= max_frames:
            break
    cap.release()


def find_plate_candidates(frame: "cv2.Mat", max_candidates: int) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 20:
            continue
        area = w * h
        if area < 1500:
            continue
        ratio = w / float(h)
        if ratio < 2.0 or ratio > 6.0:
            continue
        candidates.append((x, y, w, h))
        if len(candidates) >= max_candidates:
            break
    return candidates


def crop_with_padding(frame: "cv2.Mat", rect: Tuple[int, int, int, int]) -> "cv2.Mat":
    x, y, w, h = rect
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.20)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w + pad_x, frame.shape[1])
    y1 = min(y + h + pad_y, frame.shape[0])
    return frame[y0:y1, x0:x1]


def ocr_plate(reader: "easyocr.Reader", roi: "cv2.Mat") -> Optional[Tuple[str, float]]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    results = reader.readtext(thresh, detail=1, paragraph=False, allowlist=ALLOWLIST)
    if not results:
        return None
    best = max(results, key=lambda item: item[2])
    return best[1], float(best[2])


def record_hit(
    store: Dict[str, Dict[str, object]],
    plate: str,
    timestamp: float,
    confidence: float,
) -> None:
    entry = store.setdefault(
        plate,
        {
            "count": 0,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "confidences": [],
            "sample_timestamps": [],
        },
    )
    entry["count"] = int(entry["count"]) + 1
    entry["first_seen"] = min(float(entry["first_seen"]), timestamp)
    entry["last_seen"] = max(float(entry["last_seen"]), timestamp)
    entry["confidences"].append(confidence)
    if len(entry["sample_timestamps"]) < 5:
        entry["sample_timestamps"].append(timestamp)


def summarize_hits(store: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    summary = []
    for plate, entry in store.items():
        confidences = entry["confidences"]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        summary.append(
            {
                "plate": plate,
                "count": entry["count"],
                "first_seen": entry["first_seen"],
                "last_seen": entry["last_seen"],
                "avg_confidence": round(avg_conf, 3),
                "sample_timestamps": entry["sample_timestamps"],
            }
        )
    summary.sort(key=lambda item: (-item["count"], item["plate"]))
    return summary


def save_debug_crop(debug_dir: str, plate: str, timestamp: float, roi: "cv2.Mat") -> None:
    os.makedirs(debug_dir, exist_ok=True)
    safe_plate = re.sub(r"[^A-Z0-9]", "_", plate)
    filename = f"{safe_plate}_{timestamp:.2f}.jpg"
    path = os.path.join(debug_dir, filename)
    cv2.imwrite(path, roi)


def process_video(
    video_path: str,
    sample_fps: float,
    max_frames: Optional[int],
    min_confidence: float,
    max_candidates: int,
    strict_patterns: bool,
    use_gpu: bool,
    debug_dir: Optional[str],
) -> Dict[str, object]:
    reader = build_reader(use_gpu)
    hits: Dict[str, Dict[str, object]] = {}

    start_time = time.perf_counter()
    processed_frames = 0
    for frame_idx, timestamp, frame in iter_sampled_frames(video_path, sample_fps, max_frames):
        candidates = find_plate_candidates(frame, max_candidates)
        seen_in_frame = set()
        for rect in candidates:
            roi = crop_with_padding(frame, rect)
            ocr_result = ocr_plate(reader, roi)
            if not ocr_result:
                continue
            raw_text, confidence = ocr_result
            if confidence < min_confidence:
                continue
            plate = normalize_plate(raw_text)
            if not plate or plate in seen_in_frame:
                continue
            if not is_valid_au_plate(plate, strict_patterns=strict_patterns):
                continue
            seen_in_frame.add(plate)
            record_hit(hits, plate, timestamp, confidence)
            if debug_dir:
                save_debug_crop(debug_dir, plate, timestamp, roi)
        processed_frames += 1

    elapsed = time.perf_counter() - start_time
    summary = summarize_hits(hits)

    return {
        "video": video_path,
        "frames_processed": processed_frames,
        "sample_fps": sample_fps,
        "processing_seconds": round(elapsed, 2),
        "plates_found": len(summary),
        "plates": summary,
    }


def write_output(output_path: str, data: Dict[str, object]) -> None:
    if output_path.lower().endswith(".csv"):
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "plate",
                    "count",
                    "first_seen",
                    "last_seen",
                    "avg_confidence",
                ],
            )
            writer.writeheader()
            for row in data["plates"]:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        return

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def print_summary(data: Dict[str, object]) -> None:
    print("")
    print("Detected Australian plates")
    print("==========================")
    if not data["plates"]:
        print("No plates detected.")
        return
    for plate in data["plates"]:
        print(
            f"{plate['plate']:8} | count={plate['count']:<3} | "
            f"first={plate['first_seen']:.2f}s | "
            f"last={plate['last_seen']:.2f}s | "
            f"avg_conf={plate['avg_confidence']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan dashcam footage and list Australian registration plates."
    )
    parser.add_argument("video", help="Path to dashcam video file")
    parser.add_argument("--output", help="Write JSON or CSV results to file")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second to sample")
    parser.add_argument("--max-frames", type=int, help="Stop after processing N sampled frames")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum OCR confidence")
    parser.add_argument("--max-candidates", type=int, default=10, help="Max plate candidates per frame")
    parser.add_argument(
        "--strict-patterns",
        action="store_true",
        help="Only accept known Australian plate patterns",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR if available")
    parser.add_argument("--debug-dir", help="Save plate crops for review")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = process_video(
        video_path=args.video,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        max_candidates=args.max_candidates,
        strict_patterns=args.strict_patterns,
        use_gpu=args.gpu,
        debug_dir=args.debug_dir,
    )
    if args.output:
        write_output(args.output, data)
        print(f"Wrote results to {args.output}")
    print_summary(data)


if __name__ == "__main__":
    main()
