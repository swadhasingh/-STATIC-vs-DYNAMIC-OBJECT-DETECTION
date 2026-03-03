"""
=============================================================
  CONFIRMATORY TEST 1 — DYNAMIC CAR DETECTION
  White colour spots from frame difference (f_n - f_{n+K})

  PURPOSE:
    Confirm that ALL cars/vehicles are correctly detected as
    DYNAMIC objects using frame subtraction.
    White spots (motion pixels) inside bounding boxes prove
    the vehicle is moving.

  OUTPUT:
    - Left panel  : original frame with YOLO vehicle bboxes
    - Right panel : motion mask overlaid with white spots
    - A vehicle is CONFIRMED DYNAMIC if white-pixel ratio > threshold
    - Results saved as: test1_dynamic_cars_output.mp4
    - Per-frame CSV log: test1_dynamic_cars_log.csv

  WHAT TO LOOK FOR:
    * Green box on left  → vehicle detected
    * White blob on right inside the same region → motion confirmed
    * "DYNAMIC ✓" label on vehicle → test PASSED for that vehicle
    * "STATIC?"  label → warning, vehicle may be stopped
=============================================================
"""

import cv2
import numpy as np
import csv
import os
import argparse
import time
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
YOLO_MODEL          = "yolov8n.pt"
YOLO_CONF           = 0.40
YOLO_IOU            = 0.45
FRAME_GAP           = 8          # K: f1 vs f(1+K)
PIXEL_DIFF_THRESH   = 25         # threshold for binarising frame diff
DYNAMIC_RATIO_THRESH = 0.15      # if >15% of bbox pixels are white → DYNAMIC

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_DYNAMIC = (0,   255,   0)   # Green — dynamic confirmed
COLOR_STATIC  = (0,     0, 255)   # Red   — possibly stopped
COLOR_WHITE   = (255, 255, 255)


def compute_motion_mask(gray1, gray2, thresh=PIXEL_DIFF_THRESH):
    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


def pixel_ratio_in_box(mask, x1, y1, x2, y2):
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return cv2.countNonZero(roi) / roi.size if roi.size > 0 else 0.0


def run_test1(input_path, output_path="test1_dynamic_cars_output.mp4",
              log_path="test1_dynamic_cars_log.csv", show_preview=False):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Side-by-side output (double width)
    out_W = W * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (out_W, H))

    print(f"\n{'='*65}")
    print(f"  CONFIRMATORY TEST 1 — Dynamic Car Detection")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Log    : {log_path}")
    print(f"  Frame gap K = {FRAME_GAP}  |  Dynamic threshold = {DYNAMIC_RATIO_THRESH*100:.0f}%")
    print(f"{'='*65}\n")

    model = YOLO(YOLO_MODEL)
    print("  YOLO loaded. Running test...\n")

    frame_buffer = []
    frame_idx    = 0
    t_start      = time.time()

    total_vehicles  = 0
    dynamic_confirm = 0
    static_warn     = 0

    csv_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray)
        if len(frame_buffer) > FRAME_GAP + 1:
            frame_buffer.pop(0)

        if len(frame_buffer) == FRAME_GAP + 1:
            mask = compute_motion_mask(frame_buffer[0], frame_buffer[-1])
        else:
            mask = np.zeros((H, W), dtype=np.uint8)

        # ── Left panel: annotated original frame ──────────────────────────────
        left = frame.copy()

        # ── Right panel: motion mask as BGR with white spots ──────────────────
        # White blobs on black background = motion regions
        right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # YOLO detect vehicles only
        results = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

        frame_dyn = 0
        frame_sta = 0

        if results and results[0].boxes is not None:
            boxes     = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            coords    = boxes.xyxy.cpu().numpy().astype(int)
            names     = model.names

            for box, cid in zip(coords, class_ids):
                cls_name = names[cid].lower()
                if cls_name not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                ratio = pixel_ratio_in_box(mask, x1, y1, x2, y2)
                is_dynamic = ratio > DYNAMIC_RATIO_THRESH

                color = COLOR_DYNAMIC if is_dynamic else COLOR_STATIC
                tag   = "DYNAMIC OK" if is_dynamic else "STATIC?"

                total_vehicles += 1
                if is_dynamic:
                    dynamic_confirm += 1
                    frame_dyn += 1
                else:
                    static_warn += 1
                    frame_sta += 1

                # Draw thin bbox on left panel
                cv2.rectangle(left, (x1, y1), (x2, y2), color, 1)
                cv2.putText(left, f"{cls_name} {tag} {ratio*100:.0f}%",
                            (x1+2, max(y1-5,10)), FONT, 0.40, color, 1, cv2.LINE_AA)

                # Draw thin bbox on right panel (mask view)
                cv2.rectangle(right, (x1, y1), (x2, y2), color, 1)
                cv2.putText(right, f"{ratio*100:.0f}%",
                            (x1+2, max(y1-5,10)), FONT, 0.40, color, 1, cv2.LINE_AA)

                csv_rows.append({
                    "frame": frame_idx,
                    "class": cls_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "pixel_ratio": f"{ratio:.4f}",
                    "is_dynamic": is_dynamic,
                    "tag": tag
                })

        # Panel labels
        cv2.putText(left, "LEFT: YOLO Vehicle Detection", (8, 22),
                    FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(left, f"Frame {frame_idx} | Dyn:{frame_dyn} Sta:{frame_sta}",
                    (8, 44), FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(right, "RIGHT: Frame-Diff White Spots", (8, 22),
                    FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, f"White=Motion  |  K={FRAME_GAP} frame gap",
                    (8, 44), FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        combined = np.hstack([left, right])
        out.write(combined)

        if show_preview:
            cv2.imshow("Test 1 — Dynamic Cars [Q=quit]", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} | Dyn confirmed: {dynamic_confirm} | Warnings: {static_warn}")

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Write CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","class","x1","y1","x2","y2",
                                                "pixel_ratio","is_dynamic","tag"])
        writer.writeheader()
        writer.writerows(csv_rows)

    elapsed = time.time() - t_start
    pct_dyn = (dynamic_confirm / total_vehicles * 100) if total_vehicles > 0 else 0

    print(f"\n{'='*65}")
    print(f"  TEST 1 RESULTS — Dynamic Car Detection")
    print(f"{'='*65}")
    print(f"  Total vehicle detections : {total_vehicles:,}")
    print(f"  Confirmed DYNAMIC        : {dynamic_confirm:,}  ({pct_dyn:.1f}%)")
    print(f"  Static/stopped warnings  : {static_warn:,}  ({100-pct_dyn:.1f}%)")
    print(f"  Processing time          : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"  Output video             : {output_path}")
    print(f"  CSV log                  : {log_path}")
    verdict = "PASS" if pct_dyn >= 70.0 else "REVIEW"
    print(f"\n  TEST VERDICT : {verdict}  (>70% vehicles dynamic = PASS)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test 1: Dynamic Car Detection via Frame Diff")
    p.add_argument("--input",   "-i", required=True, help="Input video path")
    p.add_argument("--output",  "-o", default="test1_dynamic_cars_output.mp4")
    p.add_argument("--log",     "-l", default="test1_dynamic_cars_log.csv")
    p.add_argument("--preview", "-p", action="store_true")
    args = p.parse_args()
    run_test1(args.input, args.output, args.log, args.preview)