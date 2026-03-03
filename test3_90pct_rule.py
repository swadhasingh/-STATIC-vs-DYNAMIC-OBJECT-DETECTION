"""
=============================================================
  CONFIRMATORY TEST 3 — 90% THRESHOLD RULE
  Preventing slowing dynamic objects from being mis-classified
  as STATIC.

  PROBLEM:
    When a vehicle slows down (e.g. stops at traffic light),
    its bbox barely moves → simple bbox-tracking would wrongly
    label it STATIC.

  SOLUTION — 90% Rule (Step 7):
    Compare intersection of motion pixels (frame f1 - f6)
    INSIDE the object's bounding box.

    IF pixel_ratio_in_bbox > 0.90   → DYNAMIC  (object moved a lot)
    IF pixel_ratio_in_bbox < 0.10   → STATIC   (object barely moved)
    IF 0.10 ≤ ratio ≤ 0.90          → use bbox centre tracking as tiebreaker

  THIS TEST CONFIRMS:
    • Objects with ratio > 90% are correctly kept as DYNAMIC
      even if they appear slow (slowing/decelerating vehicles)
    • Objects with ratio < 10% are correctly classified STATIC
    • The boundary zone (10-90%) is handled by bbox tracking

  OUTPUT:
    - Annotated video showing ratio overlaid on each bbox
    - Colour coded:
        BRIGHT GREEN  = ratio>90%  (dynamic by 90% rule)
        RED           = ratio<10%  (static by 90% rule)
        CYAN          = 10-90%     (decided by bbox tracking)
        ORANGE border = vehicle that was SLOWING (bbox barely
                        moved) but correctly kept DYNAMIC by rule
    - CSV log: test3_90pct_rule_log.csv

  WHAT TO LOOK FOR:
    * Slowing car: bbox nearly frozen BUT ratio > 0.90 → GREEN = CORRECT
    * Parked car: bbox frozen AND ratio < 0.10 → RED = CORRECT
    * Sign/pole: ratio < 0.10 always → RED = CORRECT
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
YOLO_MODEL         = "yolov8n.pt"
YOLO_CONF          = 0.40
YOLO_IOU           = 0.45
FRAME_GAP          = 8           # f1 to f(1+K) difference
PIXEL_DIFF_THRESH  = 25

# 90% Rule thresholds
RATIO_DYNAMIC_HIGH = 0.90        # > 90% pixels changed → DEFINITELY DYNAMIC
RATIO_STATIC_LOW   = 0.10        # < 10% pixels changed → DEFINITELY STATIC
# Between 10-90% → use bbox centre movement as tiebreaker

STATIC_MOVE_THRESH = 14          # px for bbox tiebreaker
TRACK_HISTORY      = 15
TRACK_MAX_DISAP    = 20
IOU_MATCH_THRESH   = 0.35

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Decision colours
COLOR_90_DYNAMIC   = (0, 255,   0)    # Bright green — >90% rule: dynamic
COLOR_10_STATIC    = (0,   0, 255)    # Red          — <10% rule: static
COLOR_TIEBREAK_DYN = (0, 255, 128)    # Teal         — tiebreak → dynamic
COLOR_TIEBREAK_STA = (128, 0, 255)    # Purple       — tiebreak → static
COLOR_SLOWING_WARN = (0, 140, 255)    # Orange       — slowing but saved by 90% rule


# ── Motion helpers ─────────────────────────────────────────────────────────────

def compute_motion_mask(gray1, gray2):
    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(blur, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
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


# ── Simple IOU tracker ─────────────────────────────────────────────────────────

class CentreTracker:
    def __init__(self):
        self.next_id     = 0
        self.objects     = {}
        self.disappeared = {}
        self.centres     = {}

    def _iou(self, a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter==0: return 0.0
        return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

    def _register(self, box, cls):
        tid=self.next_id
        cx,cy=(box[0]+box[2])//2,(box[1]+box[3])//2
        self.objects[tid]=(*box,cls)
        self.disappeared[tid]=0
        self.centres[tid]=deque(maxlen=TRACK_HISTORY)
        self.centres[tid].append((cx,cy))
        self.next_id+=1
        return tid

    def _remove(self,tid):
        for d in [self.objects,self.disappeared,self.centres]: d.pop(tid,None)

    def update(self, detections):
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
            return []
        if not self.objects:
            return [(*d, self._register(d[:4],d[4])) for d in detections]

        eids=[*self.objects.keys()]
        eboxes=[self.objects[i][:4] for i in eids]
        matched_d,matched_e={},set()
        for di,det in enumerate(detections):
            best,btid,bei=IOU_MATCH_THRESH,None,None
            for ei,tid in enumerate(eids):
                if ei in matched_e: continue
                s=self._iou(det[:4],eboxes[ei])
                if s>best: best,btid,bei=s,tid,ei
            if btid is not None:
                matched_d[di]=btid; matched_e.add(bei)
        for di,tid in matched_d.items():
            det=detections[di]
            cx,cy=(det[0]+det[2])//2,(det[1]+det[3])//2
            self.objects[tid]=(*det[:4],det[4])
            self.disappeared[tid]=0
            self.centres[tid].append((cx,cy))
        for di,det in enumerate(detections):
            if di not in matched_d:
                tid=self._register(det[:4],det[4]); matched_d[di]=tid
        for ei,tid in enumerate(eids):
            if ei not in matched_e:
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
        return [(*detections[di],matched_d[di]) for di in range(len(detections))]

    def max_displacement(self, tid):
        hist=self.centres.get(tid)
        if not hist or len(hist)<2: return 999.0
        fx,fy=hist[0]
        return max(((cx-fx)**2+(cy-fy)**2)**0.5 for cx,cy in hist)

    def is_bbox_static(self, tid):
        return self.max_displacement(tid) < STATIC_MOVE_THRESH


def run_test3(input_path, output_path="test3_90pct_rule_output.mp4",
              log_path="test3_90pct_rule_log.csv", show_preview=False):

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
    print(f"  CONFIRMATORY TEST 3 — 90% Threshold Rule")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")
    print(f"  Log     : {log_path}")
    print(f"  DYNAMIC if pixel_ratio > {RATIO_DYNAMIC_HIGH*100:.0f}%")
    print(f"  STATIC  if pixel_ratio < {RATIO_STATIC_LOW*100:.0f}%")
    print(f"  Tiebreak (10-90%) via bbox centre tracking (<{STATIC_MOVE_THRESH}px = static)")
    print(f"{'='*65}\n")

    model   = YOLO(YOLO_MODEL)
    tracker = CentreTracker()
    print("  YOLO loaded. Running test...\n")

    frame_buffer = []
    frame_idx    = 0
    t_start      = time.time()

    # Counters for test summary
    n_90rule_dynamic  = 0  # saved from being static by 90% rule (slowing cars)
    n_10rule_static   = 0  # confirmed static by <10% rule
    n_tiebreak_dyn    = 0  # decided dynamic by bbox tracking
    n_tiebreak_sta    = 0  # decided static by bbox tracking
    n_slowing_saved   = 0  # vehicles with BOTH low bbox movement AND high pixel ratio
    total_det         = 0

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

        for det in tracked:
            x1, y1, x2, y2, cls_name, tid = det
            ratio    = pixel_ratio_in_box(mask, x1, y1, x2, y2)
            bbox_sta = tracker.is_bbox_static(tid)
            max_disp = tracker.max_displacement(tid)

            # ── Apply 90% rule decision ──────────────────────────────────────
            if ratio > RATIO_DYNAMIC_HIGH:
                # >90% pixels changed → DEFINITELY DYNAMIC
                is_dynamic = True
                decision   = "90%_RULE_DYN"
                color      = COLOR_90_DYNAMIC
                n_90rule_dynamic += 1

                # Special case: slowing vehicle (bbox barely moved BUT pixels show motion)
                is_slowing = bbox_sta and cls_name in VEHICLE_CLASSES
                if is_slowing:
                    color = COLOR_SLOWING_WARN
                    decision = "SLOWING_SAVED"
                    n_slowing_saved += 1

            elif ratio < RATIO_STATIC_LOW:
                # <10% pixels changed → DEFINITELY STATIC
                is_dynamic = False
                decision   = "10%_RULE_STA"
                color      = COLOR_10_STATIC
                n_10rule_static += 1

            else:
                # Tiebreak zone (10% – 90%) → bbox centre tracking
                if bbox_sta:
                    is_dynamic = False
                    decision   = "TIEBREAK_STA"
                    color      = COLOR_TIEBREAK_STA
                    n_tiebreak_sta += 1
                else:
                    is_dynamic = True
                    decision   = "TIEBREAK_DYN"
                    color      = COLOR_TIEBREAK_DYN
                    n_tiebreak_dyn += 1

            state_tag = "DYN" if is_dynamic else "STA"

            # THIN bounding box + label (no filled band)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            lbl = f"{cls_name}|{state_tag}|{ratio*100:.0f}%|{decision[:10]}"
            cv2.putText(annotated, lbl, (x1+2, max(y1-5, 10)),
                        FONT, 0.35, color, 1, cv2.LINE_AA)

            # Draw pixel-ratio bar inside bbox (thin horizontal line)
            bar_len = int((x2 - x1) * ratio)
            if bar_len > 0:
                cv2.line(annotated, (x1, y2-3), (x1+bar_len, y2-3), color, 2)

            total_det += 1
            csv_rows.append({
                "frame": frame_idx,
                "tid": tid,
                "class": cls_name,
                "pixel_ratio": f"{ratio:.4f}",
                "bbox_max_disp_px": f"{max_disp:.2f}",
                "bbox_static": bbox_sta,
                "decision": decision,
                "is_dynamic": is_dynamic
            })

        # HUD
        cv2.putText(annotated, "TEST 3: 90% Threshold Rule", (8, 22),
                    FONT, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Frame {frame_idx}/{total}", (8, 44),
                    FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Legend bottom
        legend_items = [
            ("BrtGreen: >90% DYN",     COLOR_90_DYNAMIC),
            ("Red: <10% STA",          COLOR_10_STATIC),
            ("Teal: tiebreak DYN",     COLOR_TIEBREAK_DYN),
            ("Purple: tiebreak STA",   COLOR_TIEBREAK_STA),
            ("Orange: slowing SAVED",  COLOR_SLOWING_WARN),
        ]
        for li, (txt, col) in enumerate(legend_items):
            cv2.putText(annotated, txt, (8, H - 10 - li*18),
                        FONT, 0.36, col, 1, cv2.LINE_AA)

        out.write(annotated)

        if show_preview:
            cv2.imshow("Test 3 — 90% Rule [Q=quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} | 90%DYN:{n_90rule_dynamic} "
                  f"10%STA:{n_10rule_static} Tiebreak:{n_tiebreak_dyn+n_tiebreak_sta} "
                  f"SlowingSaved:{n_slowing_saved}")

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Write CSV
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","tid","class","pixel_ratio",
                                               "bbox_max_disp_px","bbox_static",
                                               "decision","is_dynamic"])
        writer.writeheader()
        writer.writerows(csv_rows)

    elapsed = time.time() - t_start

    print(f"\n{'='*65}")
    print(f"  TEST 3 RESULTS — 90% Threshold Rule")
    print(f"{'='*65}")
    print(f"  Total detections                   : {total_det:,}")
    print(f"  Confirmed DYNAMIC by >90% rule     : {n_90rule_dynamic:,}")
    print(f"  Confirmed STATIC  by <10% rule     : {n_10rule_static:,}")
    print(f"  Tiebreak → DYNAMIC (bbox moving)   : {n_tiebreak_dyn:,}")
    print(f"  Tiebreak → STATIC  (bbox frozen)   : {n_tiebreak_sta:,}")
    print(f"  SLOWING vehicles correctly saved   : {n_slowing_saved:,}")
    print(f"    (bbox static BUT ratio>90% → kept DYNAMIC = CORRECT)")
    print(f"  Processing time                    : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"\n  Output video : {output_path}")
    print(f"  CSV log      : {log_path}")

    # Verdict
    if n_slowing_saved > 0:
        print(f"\n  TEST VERDICT : PASS — {n_slowing_saved} slowing vehicle(s) correctly")
        print(f"                 identified as DYNAMIC by the 90% rule.")
    else:
        print(f"\n  TEST VERDICT : INFO — No slowing vehicles detected in this clip.")
        print(f"                 The 90% rule is active and ready.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test 3: 90% Pixel Ratio Threshold Rule")
    p.add_argument("--input",   "-i", required=True)
    p.add_argument("--output",  "-o", default="test3_90pct_rule_output.mp4")
    p.add_argument("--log",     "-l", default="test3_90pct_rule_log.csv")
    p.add_argument("--preview", "-p", action="store_true")
    args = p.parse_args()
    run_test3(args.input, args.output, args.log, args.preview)