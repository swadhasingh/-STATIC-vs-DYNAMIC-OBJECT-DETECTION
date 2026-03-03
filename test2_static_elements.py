"""
=============================================================
  CONFIRMATORY TEST 2 — STATIC ELEMENT DETECTION
  Objects whose centre-of-mass movement is less than a
  defined pixel threshold over a window of frames.

  PURPOSE:
    Detect and confirm ALL static objects in the scene —
    road signs, poles, benches, parked vehicles, banners.
    Uses bbox centre tracking: if the centre moves < THRESH
    pixels over HISTORY frames → labelled STATIC.

  METHOD:
    - Track bbox centres across frames using IOU tracker
    - Compute max displacement over a rolling window
    - If max displacement < STATIC_MOVE_THRESH → STATIC
    - Overlay RED bounding box for static objects
    - BLUE for objects whose static status is being evaluated

  OUTPUT:
    - Annotated video: test2_static_elements_output.mp4
    - CSV log of static objects: test2_static_elements_log.csv
    - Console summary table

  WHAT TO LOOK FOR:
    RED boxes = confirmed static (movement < threshold)
    YELLOW boxes = borderline / being evaluated
    GREEN boxes = confirmed dynamic (movement ≥ threshold)
=============================================================
"""

import cv2
import numpy as np
import csv
import os
import argparse
import time
from collections import deque, defaultdict
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
YOLO_MODEL          = "yolov8n.pt"
YOLO_CONF           = 0.40
YOLO_IOU            = 0.45

STATIC_MOVE_THRESH  = 14        # px — max displacement to be declared static
TRACK_HISTORY       = 20        # frames of centre history for movement calc
TRACK_MAX_DISAP     = 20        # frames before track dropped
IOU_MATCH_THRESH    = 0.35      # min IOU to match detection to existing track
MIN_FRAMES_TO_EVAL  = 8         # need at least this many frames before deciding

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_STATIC    = (0,   0,   255)   # Red    — confirmed static
COLOR_DYNAMIC   = (0, 255,     0)   # Green  — confirmed dynamic
COLOR_EVAL      = (0, 200,   255)   # Yellow — evaluating (not enough history)


# ── Simple IOU tracker for this test ─────────────────────────────────────────
class CentreTracker:
    def __init__(self):
        self.next_id     = 0
        self.objects     = {}
        self.disappeared = {}
        self.centres     = {}   # tid → deque of (cx, cy)

    def _iou(self, a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

    def _register(self, box, cls):
        tid = self.next_id
        cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
        self.objects[tid]     = (*box, cls)
        self.disappeared[tid] = 0
        self.centres[tid]     = deque(maxlen=TRACK_HISTORY)
        self.centres[tid].append((cx, cy))
        self.next_id += 1
        return tid

    def _remove(self, tid):
        self.objects.pop(tid, None)
        self.disappeared.pop(tid, None)
        self.centres.pop(tid, None)

    def update(self, detections):
        """detections: list of (x1,y1,x2,y2,cls). Returns list of (x1,y1,x2,y2,cls,tid)."""
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > TRACK_MAX_DISAP:
                    self._remove(tid)
            return []

        if not self.objects:
            return [(*d, self._register(d[:4], d[4])) for d in detections]

        eids   = list(self.objects.keys())
        eboxes = [self.objects[i][:4] for i in eids]

        matched_d = {}
        matched_e = set()

        for di, det in enumerate(detections):
            best, best_tid, best_ei = IOU_MATCH_THRESH, None, None
            for ei, tid in enumerate(eids):
                if ei in matched_e: continue
                s = self._iou(det[:4], eboxes[ei])
                if s > best:
                    best, best_tid, best_ei = s, tid, ei
            if best_tid is not None:
                matched_d[di] = best_tid
                matched_e.add(best_ei)

        for di, tid in matched_d.items():
            det = detections[di]
            cx, cy = (det[0]+det[2])//2, (det[1]+det[3])//2
            self.objects[tid]     = (*det[:4], det[4])
            self.disappeared[tid] = 0
            self.centres[tid].append((cx, cy))

        for di, det in enumerate(detections):
            if di not in matched_d:
                tid = self._register(det[:4], det[4])
                matched_d[di] = tid

        for ei, tid in enumerate(eids):
            if ei not in matched_e:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > TRACK_MAX_DISAP:
                    self._remove(tid)

        return [(*detections[di], matched_d[di]) for di in range(len(detections))]

    def max_displacement(self, tid):
        """Maximum displacement of centre from its first recorded position."""
        hist = self.centres.get(tid)
        if not hist or len(hist) < 2:
            return 999.0
        fx, fy = hist[0]
        return max(((cx-fx)**2 + (cy-fy)**2)**0.5 for cx, cy in hist)

    def history_len(self, tid):
        return len(self.centres.get(tid, []))


def run_test2(input_path, output_path="test2_static_elements_output.mp4",
              log_path="test2_static_elements_log.csv", show_preview=False):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print(f"\n{'='*65}")
    print(f"  CONFIRMATORY TEST 2 — Static Element Detection")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Log    : {log_path}")
    print(f"  Static threshold : {STATIC_MOVE_THRESH}px  |  History : {TRACK_HISTORY} frames")
    print(f"{'='*65}\n")

    model   = YOLO(YOLO_MODEL)
    tracker = CentreTracker()
    print("  YOLO loaded. Running test...\n")

    frame_idx         = 0
    t_start           = time.time()
    total_detections  = 0
    static_confirmed  = 0
    dynamic_confirmed = 0
    eval_count        = 0

    # Track which IDs were ever confirmed static (for summary)
    ever_static = {}   # tid → (cls, max_disp)

    csv_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        results = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        raw = []
        if results and results[0].boxes is not None:
            boxes     = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            coords    = boxes.xyxy.cpu().numpy().astype(int)
            names     = model.names
            for box, cid in zip(coords, class_ids):
                raw.append((*box.tolist(), names[cid].lower()))

        tracked = tracker.update(raw)

        frame_static  = 0
        frame_dynamic = 0

        for det in tracked:
            x1, y1, x2, y2, cls_name, tid = det
            cx, cy   = (x1+x2)//2, (y1+y2)//2
            hist_len = tracker.history_len(tid)
            max_disp = tracker.max_displacement(tid)

            if hist_len < MIN_FRAMES_TO_EVAL:
                # Not enough history yet
                color = COLOR_EVAL
                tag   = f"EVAL({hist_len}f)"
                state = "evaluating"
                eval_count += 1
            elif max_disp < STATIC_MOVE_THRESH:
                color = COLOR_STATIC
                tag   = f"STATIC {max_disp:.1f}px"
                state = "static"
                static_confirmed += 1
                frame_static += 1
                ever_static[tid] = (cls_name, max_disp)
            else:
                color = COLOR_DYNAMIC
                tag   = f"DYN {max_disp:.1f}px"
                state = "dynamic"
                dynamic_confirmed += 1
                frame_dynamic += 1

            # THIN bounding box, no filled band
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(annotated, f"{cls_name}|{tag}", (x1+2, max(y1-5, 10)),
                        FONT, 0.38, color, 1, cv2.LINE_AA)
            # Draw centre dot
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            total_detections += 1
            csv_rows.append({
                "frame": frame_idx,
                "tid": tid,
                "class": cls_name,
                "cx": cx, "cy": cy,
                "max_displacement_px": f"{max_disp:.2f}",
                "history_frames": hist_len,
                "state": state
            })

        # HUD
        cv2.putText(annotated, "TEST 2: Static Element Detection", (8, 22),
                    FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Frame {frame_idx} | STA:{frame_static} DYN:{frame_dynamic}",
                    (8, 44), FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Threshold: <{STATIC_MOVE_THRESH}px = STATIC",
                    (8, 64), FONT, 0.44, (100, 255, 100), 1, cv2.LINE_AA)
        # Legend
        cv2.putText(annotated, "RED=Static  GREEN=Dynamic  YELLOW=Evaluating",
                    (8, H-12), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(annotated)

        if show_preview:
            cv2.imshow("Test 2 — Static Elements [Q=quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} | Static: {static_confirmed} | Dynamic: {dynamic_confirmed}")

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Write CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","tid","class","cx","cy",
                                               "max_displacement_px","history_frames","state"])
        writer.writeheader()
        writer.writerows(csv_rows)

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print(f"  TEST 2 RESULTS — Static Element Detection")
    print(f"{'='*65}")
    print(f"  Total detections     : {total_detections:,}")
    print(f"  Static confirmed     : {static_confirmed:,}")
    print(f"  Dynamic confirmed    : {dynamic_confirmed:,}")
    print(f"  Still evaluating     : {eval_count:,}")
    print(f"  Processing time      : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"\n  CONFIRMED STATIC OBJECTS (unique track IDs):")
    print(f"  {'TID':<6} {'CLASS':<20} {'MAX DISP (px)':<15}")
    print(f"  {'─'*6} {'─'*20} {'─'*15}")
    for tid, (cls, disp) in sorted(ever_static.items()):
        print(f"  {tid:<6} {cls:<20} {disp:<15.2f}")
    if not ever_static:
        print("  (none confirmed in this clip)")
    print(f"\n  Output video : {output_path}")
    print(f"  CSV log      : {log_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test 2: Static Element Detection via Movement Threshold")
    p.add_argument("--input",   "-i", required=True)
    p.add_argument("--output",  "-o", default="test2_static_elements_output.mp4")
    p.add_argument("--log",     "-l", default="test2_static_elements_log.csv")
    p.add_argument("--preview", "-p", action="store_true")
    args = p.parse_args()
    run_test2(args.input, args.output, args.log, args.preview)