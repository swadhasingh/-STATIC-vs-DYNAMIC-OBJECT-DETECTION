"""
=============================================================
  STATIC vs DYNAMIC OBJECT DETECTION — YOLO + BBOX TRACKING
  Steps 1-7 Combined Pipeline
  Author  : Dr Ninad NMIMS Assignment
  Version : 5.0

  PIPELINE STEPS:
  ┌──────────────────────────────────────────────────────┐
  │  Step 1 : Load video locally                         │
  │  Step 2 : Crop end-credits (last N seconds)          │
  │  Step 3 : TinyYOLO / YOLOv8n real-time detection     │
  │  Step 4 : Count objects per frame                    │
  │  Step 5 : Frame subtraction → motion mask            │
  │  Step 6 : Union of frame-diff & bbox → static/dyn    │
  │  Step 7 : % overlap (90% rule) threshold decision    │
  │                                                      │
  │  VERTICAL divider  (1 line, centre of frame)         │
  │    Lane 1 LEFT  = cars going AWAY from camera        │
  │    Lane 2 RIGHT = cars coming TOWARD camera          │
  │    Each unique track ID counted ONCE per lane        │
  │                                                      │
  │  HORIZONTAL zones  (4 rows, top → bottom)            │
  │    Zone 1 = far  (small bbox, fast)                  │
  │    Zone 4 = near (large bbox, slow / stopped)        │
  │                                                      │
  │  STATIC detection via DUAL METHOD:                   │
  │    1) BBox tracking: centre movement < threshold     │
  │    2) Frame-diff pixel ratio (90% rule):             │
  │       if >90% pixels in bbox changed → DYNAMIC       │
  │       if <10% pixels in bbox changed → STATIC        │
  │    VEHICLES are ALWAYS green — never labelled static  │
  │                                                      │
  │  BOUNDING BOXES:                                     │
  │    Thin (thickness=1) green lines only               │
  │    NO filled label band above box                    │
  │    Small text label floated just above box           │
  └──────────────────────────────────────────────────────┘
=============================================================
"""

import cv2
import numpy as np
import argparse
import os
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CREDITS_SECONDS       = 10
FRAME_GAP             = 8        # K: frames apart for pixel-diff mask (f1 vs f6 style)
PIXEL_DIFF_THRESH     = 25       # binarisation threshold for motion mask

# Static detection (dual method for non-vehicles)
STATIC_MOVE_THRESH    = 14       # px — max bbox centre movement over history
STATIC_PIXEL_RATIO    = 0.10     # if <10% pixels changed inside bbox → static
                                  # (90% rule: >90% changed = dynamic)

# Tracker
TRACK_HISTORY         = 15       # frames of centre history to keep
TRACK_MAX_DISAPPEARED = 20       # frames before a track is dropped

# Speed check — horizontal zones
NUM_H_ZONES           = 4        # horizontal speed zones
SPEED_MIN_FRAMES      = 35       # if car crosses all zones in fewer frames → SPEEDING

YOLO_CONF             = 0.40
YOLO_IOU              = 0.45
YOLO_MODEL            = "yolov8n.pt"

# ── Colours (BGR) ─────────────────────────────
COLOR_DYNAMIC   = (0,   255,   0)   # Green  — dynamic non-vehicle
COLOR_STATIC    = (0,     0, 255)   # Red    — static non-vehicle
COLOR_VEHICLE   = (0,   255,   0)   # Green  — all vehicles (always)
COLOR_SPEEDING  = (0,   140, 255)   # Orange — speeding vehicle
COLOR_LANE_V    = (0,   255, 255)   # Cyan   — vertical lane divider
COLOR_ZONE_H    = (255, 255,   0)   # Yellow — horizontal zone lines
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Vehicle classes — ALWAYS green, counted in lane stats
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}

# ── Known STATIC infrastructure objects (road banners, signs, etc.) ──────────
# These classes are treated as STATIC regardless of pixel-diff result.
ALWAYS_STATIC_CLASSES = {"stop sign", "traffic light", "parking meter", "bench"}


# ─────────────────────────────────────────────
#  TERMINAL HELPERS
# ─────────────────────────────────────────────

def sec_to_hms(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}"

def print_banner(title):
    w = 70
    print("\n" + "=" * w)
    pad = (w - len(title) - 2) // 2
    print(f"{' '*pad} {title}")
    print("=" * w)

def print_section(title):
    print(f"\n  +-- {title} " + "-" * max(0, 60 - len(title)) + "+")

def print_row(label, value):
    print(f"  |  {label:<44} {value}")

def print_bar(label, count, total, bar_width=26):
    pct    = (count / total * 100) if total > 0 else 0
    filled = int(bar_width * count / total) if total > 0 else 0
    bar    = "#" * filled + "." * (bar_width - filled)
    print(f"  |  {label:<16} [{bar}]  {count:>6,} vehicles  ({pct:5.1f}%)")


# ─────────────────────────────────────────────
#  SIMPLE IOU TRACKER
# ─────────────────────────────────────────────

class SimpleTracker:
    """
    Lightweight IOU tracker.
    - Assigns each detection a persistent track ID.
    - Stores bbox centre history (for static detection).
    - Stores horizontal zone history (for speed detection).
    - Tracks which vertical lane a vehicle entered (for unique counting).
    """

    def __init__(self):
        self.next_id          = 0
        self.objects          = {}   # tid → {x1,y1,x2,y2,cx,cy,cls}
        self.disappeared      = {}   # tid → int
        self.centre_hist      = {}   # tid → deque[(cx,cy)]
        self.zone_hist        = {}   # tid → deque[zone_int]
        self.lane_logged      = {}   # tid → set of lane numbers already counted

    @staticmethod
    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(self, detections):
        """
        detections : list of (x1,y1,x2,y2,cls_name)
        returns     : list of (x1,y1,x2,y2,cls_name,tid)
        """
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > TRACK_MAX_DISAPPEARED:
                    self._remove(tid)
            return []

        if not self.objects:
            for det in detections:
                self._register(det)
            ids = list(range(self.next_id - len(detections), self.next_id))
            return [(*d, i) for d, i in zip(detections, ids)]

        eids   = list(self.objects.keys())
        eboxes = [[self.objects[i]["x1"], self.objects[i]["y1"],
                   self.objects[i]["x2"], self.objects[i]["y2"]] for i in eids]

        matched_det   = {}
        matched_exist = set()

        for di, det in enumerate(detections):
            best_iou, best_tid, best_ei = 0.35, None, None
            for ei, tid in enumerate(eids):
                if ei in matched_exist: continue
                s = self._iou(det[:4], eboxes[ei])
                if s > best_iou:
                    best_iou, best_tid, best_ei = s, tid, ei
            if best_tid is not None:
                matched_det[di]  = best_tid
                matched_exist.add(best_ei)

        for di, tid in matched_det.items():
            det = detections[di]
            cx, cy = (det[0]+det[2])//2, (det[1]+det[3])//2
            self.objects[tid] = {"x1":det[0],"y1":det[1],"x2":det[2],"y2":det[3],
                                  "cx":cx,"cy":cy,"cls":det[4]}
            self.centre_hist[tid].append((cx, cy))
            self.disappeared[tid] = 0

        for di, det in enumerate(detections):
            if di not in matched_det:
                tid = self._register(det)
                matched_det[di] = tid

        for ei, tid in enumerate(eids):
            if ei not in matched_exist:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > TRACK_MAX_DISAPPEARED:
                    self._remove(tid)

        return [(*detections[di], matched_det[di]) for di in range(len(detections))]

    def _register(self, det):
        tid = self.next_id
        cx, cy = (det[0]+det[2])//2, (det[1]+det[3])//2
        self.objects[tid]     = {"x1":det[0],"y1":det[1],"x2":det[2],"y2":det[3],
                                  "cx":cx,"cy":cy,"cls":det[4]}
        self.disappeared[tid] = 0
        self.centre_hist[tid] = deque(maxlen=TRACK_HISTORY)
        self.zone_hist[tid]   = deque(maxlen=SPEED_MIN_FRAMES + 10)
        self.lane_logged[tid] = set()
        self.centre_hist[tid].append((cx, cy))
        self.next_id += 1
        return tid

    def _remove(self, tid):
        for d in [self.objects, self.disappeared, self.centre_hist,
                  self.zone_hist, self.lane_logged]:
            d.pop(tid, None)

    # ── Static / speed queries ──────────────────────────────────────────────

    def pixel_movement(self, track_id):
        """Max displacement of bbox centre over stored history (pixels)."""
        hist = self.centre_hist.get(track_id)
        if not hist or len(hist) < 2:
            return 999.0
        fx, fy = hist[0]
        return max(((cx-fx)**2+(cy-fy)**2)**0.5 for cx,cy in hist)

    def is_bbox_static(self, track_id):
        """True if bbox barely moved (non-vehicle static check)."""
        return self.pixel_movement(track_id) < STATIC_MOVE_THRESH

    def update_zone(self, tid, zone):
        if tid in self.zone_hist:
            self.zone_hist[tid].append(zone)

    def is_speeding(self, tid):
        """
        True if vehicle crossed from zone 4 (bottom/near) to zone 1 (top/far)
        in fewer than SPEED_MIN_FRAMES.
        """
        zh = list(self.zone_hist.get(tid, []))
        if len(zh) < 4:
            return False
        try:
            last_z4  = len(zh) - 1 - zh[::-1].index(4)
            first_z1 = next((i for i,z in enumerate(zh) if z==1 and i>last_z4), None)
            if first_z1 is not None and (first_z1 - last_z4) < SPEED_MIN_FRAMES:
                return True
        except ValueError:
            pass
        return False

    def try_count_lane(self, tid, lane):
        """Returns True (and records) the FIRST time this track is in this lane."""
        if tid not in self.lane_logged:
            return False
        if lane not in self.lane_logged[tid]:
            self.lane_logged[tid].add(lane)
            return True
        return False


# ─────────────────────────────────────────────
#  VISION HELPERS
# ─────────────────────────────────────────────

def get_video_info(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return total, fps, w, h


# ── STEP 5: Frame subtraction motion mask ─────────────────────────────────────
def compute_motion_mask(gray1, gray2):
    """
    Step 5 — Frame difference mask.
    Binary motion mask: white pixels = pixels that changed significantly.
    Uses |f1 - f(1+K)| with Gaussian blur + morphological cleanup.
    """
    diff = cv2.absdiff(gray1, gray2)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(blur, PIXEL_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


# ── STEP 7: % overlap (90% rule) ─────────────────────────────────────────────
def pixel_ratio_in_box(mask, x1, y1, x2, y2):
    """
    Step 7 — Fraction of pixels inside bbox that are active in the motion mask.

    90% Rule:
      ratio > 0.90  → DYNAMIC  (>90% of pixels in bbox changed between f1 and f6)
      ratio < 0.10  → STATIC   (<10% of pixels changed)
      in between    → use bbox centre tracking to decide
    
    This correctly handles slowing-down vehicles: even if the vehicle slows,
    if <90% of its bbox area shows motion intersection → it's still dynamic
    unless it truly stops (bbox movement also freezes).
    """
    h, w = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / roi.size


def get_vertical_lane(cx, frame_width):
    """Left half = Lane 1 (going away), Right half = Lane 2 (coming toward)."""
    return 1 if cx < frame_width // 2 else 2


def get_h_zone(cy, frame_height, num_zones=NUM_H_ZONES):
    """Top = Zone 1 (far/fast), Bottom = Zone N (near/slow)."""
    return min(int(cy // (frame_height / num_zones)) + 1, num_zones)


def draw_grid(frame, W, H):
    """
    Draw overlay grid:
      - 1 vertical CYAN line at centre  (lane divider)
      - 3 horizontal YELLOW dashed lines (speed zones)
    """
    cv2.line(frame, (W//2, 0), (W//2, H), COLOR_LANE_V, 2)
    cv2.putText(frame, "Lane 1 (Going Away)",
                (10, H - 12), FONT, 0.55, COLOR_LANE_V, 1, cv2.LINE_AA)
    cv2.putText(frame, "Lane 2 (Coming Close)",
                (W//2 + 8, H - 12), FONT, 0.55, COLOR_LANE_V, 1, cv2.LINE_AA)

    zone_h = H // NUM_H_ZONES
    zone_labels = {1: "Z1-Far/Fast", 2: "Z2", 3: "Z3", 4: "Z4-Near/Slow"}
    for i in range(1, NUM_H_ZONES):
        y = i * zone_h
        for x in range(0, W, 30):
            cv2.line(frame, (x, y), (min(x+15, W), y), COLOR_ZONE_H, 1)
    for z in range(1, NUM_H_ZONES + 1):
        y_mid = (z - 1) * zone_h + zone_h // 2
        cv2.putText(frame, zone_labels[z], (6, y_mid),
                    FONT, 0.42, COLOR_ZONE_H, 1, cv2.LINE_AA)


def draw_label_clean(frame, label, x1, y1, color):
    """
    Draw label WITHOUT a filled background band.
    Just thin coloured text floated slightly above the bounding box.
    """
    cv2.putText(frame, label, (x1 + 2, max(y1 - 5, 10)),
                FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_bbox(frame, x1, y1, x2, y2, color):
    """
    Draw THIN (thickness=1) bounding box with no fill/band.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


# ─────────────────────────────────────────────
#  MAIN PIPELINE  (Steps 1-7 combined)
# ─────────────────────────────────────────────

def process_video(input_path, output_path, show_preview=False,
                  credits_sec=CREDITS_SECONDS, frame_gap=FRAME_GAP):

    # ── STEP 1: Load video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total_frames, fps, W, H = get_video_info(cap)

    # ── STEP 2: Crop end-credits ──────────────────────────────────────────────
    credits_frames = int(fps * credits_sec)
    cropped_frames = max(total_frames - credits_frames, 0)
    orig_dur  = total_frames  / fps
    crop_dur  = cropped_frames / fps

    # ── STARTUP REPORT ────────────────────────────────────────────────────────
    print_banner("TRAFFIC ANALYSER  v5.0  —  NMIMS")

    print_section("VIDEO INFO")
    print_row("Input",           os.path.basename(input_path))
    print_row("Output",          os.path.basename(output_path))
    print_row("Resolution",      f"{W} x {H} px")
    print_row("FPS",             f"{fps:.2f}")

    print_section("FRAME COUNTS")
    print_row("Original total frames",
              f"{total_frames:>10,}   [ {sec_to_hms(orig_dur)} / {orig_dur:.1f}s ]")
    print_row("End-credits removed",
              f"{credits_frames:>10,}   [ {credits_sec:.1f}s ]")
    print_row("Cropped frames to process",
              f"{cropped_frames:>10,}   [ {sec_to_hms(crop_dur)} / {crop_dur:.1f}s ]")

    print_section("DETECTION CONFIG")
    print_row("YOLO model",            YOLO_MODEL)
    print_row("Static method",         "DUAL: bbox movement + 90% pixel-diff rule")
    print_row("90% rule",              ">90% pixels changed in bbox=DYNAMIC, <10%=STATIC")
    print_row("Road banners/signs",    "ALWAYS STATIC (infrastructure class override)")
    print_row("Bbox static threshold", f"< {STATIC_MOVE_THRESH}px movement = STATIC")
    print_row("Frame gap K",           str(frame_gap))
    print_row("Vertical lanes",        "2  (Left = going away, Right = coming close)")
    print_row("Horizontal speed zones",str(NUM_H_ZONES))
    print_row("Speeding threshold",    f"< {SPEED_MIN_FRAMES} frames to cross all zones")
    print_row("Bounding box style",    "Thin green lines, no label band")
    print()

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print("  Loading YOLO model ...")
    model   = YOLO(YOLO_MODEL)
    tracker = SimpleTracker()
    print("  YOLO ready.\n")
    print("  " + "-" * 66)
    print("  PROCESSING FRAMES  (Steps 3-7 active)")
    print("  " + "-" * 66)

    # ── Stats ─────────────────────────────────────────────────────────────────
    lane_counts      = {1: 0, 2: 0}
    speeding_ids     = set()
    per_class        = defaultdict(int)
    total_det        = 0
    total_static_det = 0
    total_dyn_det    = 0

    frame_buffer = []
    frame_idx    = 0
    t_start      = time.time()

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 2: credits cut-off
        if frame_idx >= cropped_frames:
            print(f"\n  [CUT] Credits boundary at frame {frame_idx}.")
            break

        # ── STEP 5: Frame subtraction ─────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray)
        if len(frame_buffer) > frame_gap + 1:
            frame_buffer.pop(0)

        if len(frame_buffer) == frame_gap + 1:
            mask = compute_motion_mask(frame_buffer[0], frame_buffer[-1])
        else:
            mask = np.zeros((H, W), dtype=np.uint8)

        annotated = frame.copy()
        draw_grid(annotated, W, H)

        # ── STEP 3: YOLO detection ────────────────────────────────────────────
        results = model.predict(source=frame, conf=YOLO_CONF,
                                iou=YOLO_IOU, verbose=False)
        raw = []
        if results and results[0].boxes is not None:
            boxes     = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            coords    = boxes.xyxy.cpu().numpy().astype(int)
            names     = model.names
            for box, cid in zip(coords, class_ids):
                raw.append((*box, names[cid].lower()))

        tracked = tracker.update(raw)

        # ── STEP 4: Count objects per frame ──────────────────────────────────
        frame_veh_count = 0

        for det in tracked:
            x1, y1, x2, y2, cls_name, tid = det
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── STEP 6+7: Union of diff & bbox + 90% rule ────────────────────
            ratio      = pixel_ratio_in_box(mask, x1, y1, x2, y2)
            is_vehicle = cls_name in VEHICLE_CLASSES

            # Road banners, signs, poles → ALWAYS STATIC (infrastructure)
            is_infrastructure = cls_name in ALWAYS_STATIC_CLASSES

            if is_vehicle:
                # ── Vehicles: ALWAYS green (dynamic) ─────────────────────────
                is_dynamic  = True
                is_speeding = False

                zone = get_h_zone(cy, H)
                tracker.update_zone(tid, zone)
                if tracker.is_speeding(tid):
                    speeding_ids.add(tid)
                    is_speeding = True

                lane = get_vertical_lane(cx, W)
                if tracker.try_count_lane(tid, lane):
                    lane_counts[lane] += 1

                frame_veh_count += 1

                color     = COLOR_SPEEDING if is_speeding else COLOR_VEHICLE
                speed_tag = " SPD!" if is_speeding else ""
                lbl = f"{cls_name} L{lane}Z{zone}{speed_tag} #{tid}"

            elif is_infrastructure:
                # ── Infrastructure: ALWAYS STATIC (road banner fix) ──────────
                is_dynamic = False
                color = COLOR_STATIC
                lbl   = f"{cls_name}|STA #{tid}"

            else:
                # ── Non-vehicles: dual static check (Step 6 + Step 7) ────────
                # 90% rule (pixel diff intersection) — Step 7 threshold
                if ratio > 0.90:
                    is_dynamic = True    # >90% of bbox pixels changed → DYNAMIC
                elif ratio < STATIC_PIXEL_RATIO:
                    is_dynamic = False   # <10% pixels changed → STATIC
                else:
                    # Middle ground → use bbox centre tracking (Step 6)
                    is_dynamic = not tracker.is_bbox_static(tid)

                color = COLOR_DYNAMIC if is_dynamic else COLOR_STATIC
                tag   = "DYN" if is_dynamic else "STA"
                lbl   = f"{cls_name}|{tag}|{ratio*100:.0f}% #{tid}"

            # ── THIN GREEN BOUNDING BOX — no label band ───────────────────────
            draw_bbox(annotated, x1, y1, x2, y2, color)
            draw_label_clean(annotated, lbl, x1, y1, color)

            # Global stats
            total_det += 1
            per_class[cls_name] += 1
            if is_vehicle or (not is_vehicle and is_dynamic):
                total_dyn_det    += 1
            else:
                total_static_det += 1

        # ── HUD (top-right) ───────────────────────────────────────────────────
        elapsed_now = time.time() - t_start
        fps_live    = frame_idx / elapsed_now if elapsed_now > 0 else 0.0
        hud = [
            f"Frame : {frame_idx:,} / {cropped_frames:,}",
            f"Vehicles : {frame_veh_count}",
            f"Speed : {fps_live:.1f} fps",
        ]
        for li, txt in enumerate(hud):
            tw = cv2.getTextSize(txt, FONT, 0.62, 2)[0][0]
            cv2.putText(annotated, txt, (W - tw - 10, 28 + li*26),
                        FONT, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

        # ── Lane counter HUD (bottom-right) ──────────────────────────────────
        hud2 = [
            "-- UNIQUE CARS --",
            f"Lane 1 (away) : {lane_counts[1]:,}",
            f"Lane 2 (close): {lane_counts[2]:,}",
            f"Speeding IDs  : {len(speeding_ids)}",
        ]
        for li, txt in enumerate(hud2):
            cv2.putText(annotated, txt,
                        (W - 260, H - 90 + li * 22),
                        FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        out.write(annotated)

        if show_preview:
            cv2.imshow("Traffic Analyser  [Q=quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n  Preview closed.")
                break

        # Terminal progress
        if frame_idx % 150 == 0 and frame_idx > 0:
            elapsed = time.time() - t_start
            fps_now = frame_idx / elapsed
            eta     = (cropped_frames - frame_idx) / max(fps_now, 0.01)
            pct     = frame_idx / cropped_frames * 100
            fill    = int(30 * frame_idx / cropped_frames)
            bar     = "#" * fill + "." * (30 - fill)
            print(f"  [{bar}] {pct:5.1f}%  |  "
                  f"{frame_idx:,}/{cropped_frames:,}  |  "
                  f"{fps_now:.1f} fps  |  ETA {sec_to_hms(eta)}")

        frame_idx += 1

    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start

    # ─────────────────────────────────────────────
    #  FINAL REPORT
    # ─────────────────────────────────────────────
    print_banner("FINAL REPORT")

    print_section("TIMING")
    print_row("Total processing time", sec_to_hms(total_time))
    print_row("Average speed",         f"{frame_idx/total_time:.2f} fps")

    print_section("FRAME SUMMARY")
    print_row("Original total frames",
              f"{total_frames:>10,}   [ {sec_to_hms(orig_dur)} / {orig_dur:.1f}s ]")
    print_row("Frames removed (credits)",
              f"{credits_frames:>10,}   [ {credits_sec:.1f}s ]")
    print_row("Cropped frames",
              f"{cropped_frames:>10,}   [ {sec_to_hms(crop_dur)} / {crop_dur:.1f}s ]")
    print_row("Frames actually processed",
              f"{frame_idx:>10,}")

    print_section("DETECTION SUMMARY")
    print_row("Total detections",        f"{total_det:,}")
    print_row("  -> DYNAMIC / vehicles", f"{total_dyn_det:,}")
    print_row("  -> STATIC",             f"{total_static_det:,}")

    print(f"\n  |  {'CLASS':<22} {'DETECTIONS':>10}")
    print(f"  |  {'─'*22}   {'─'*10}")
    for cls, cnt in sorted(per_class.items(), key=lambda x: -x[1]):
        mk = ">> " if cls in VEHICLE_CLASSES else "   "
        print(f"  |  {mk}{cls:<20} {cnt:>10,}")
    print(f"  |  (>> = vehicle)")

    print_section("LANE COUNT  (unique vehicles counted ONCE per lane via centre-of-mass)")
    total_unique = lane_counts[1] + lane_counts[2]
    print_bar("Lane 1 (Away) ", lane_counts[1], total_unique or 1)
    print_bar("Lane 2 (Close)", lane_counts[2], total_unique or 1)
    print(f"  |")
    print(f"  |  Total unique vehicles counted : {total_unique:,}")
    busiest = 1 if lane_counts[1] >= lane_counts[2] else 2
    print(f"  |  Busiest lane : Lane {busiest}  ({lane_counts[busiest]:,} vehicles)")

    print_section("SPEED / RULE VIOLATIONS")
    print_row("Vehicles flagged as SPEEDING", f"{len(speeding_ids)}")
    if speeding_ids:
        ids_str = ", ".join(str(i) for i in sorted(list(speeding_ids))[:20])
        suffix  = " ..." if len(speeding_ids) > 20 else ""
        print_row("Speeding track IDs (first 20)", ids_str + suffix)
    print_row("Speed rule",
              f"Car crosses all {NUM_H_ZONES} zones in < {SPEED_MIN_FRAMES} frames = SPEEDING")

    print_section("OUTPUT")
    print_row("Annotated video", output_path)
    print("\n" + "=" * 70 + "\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Traffic Analyser v5.0 — Steps 1-7 Combined Pipeline"
    )
    p.add_argument("--input",         "-i", required=True,
                   help="Path to input video file")
    p.add_argument("--output",        "-o", default="output_annotated.mp4",
                   help="Path to output annotated video")
    p.add_argument("--preview",       "-p", action="store_true",
                   help="Show live preview window while processing")
    p.add_argument("--credits",       "-c", type=float, default=CREDITS_SECONDS,
                   help=f"Seconds to crop from end (default {CREDITS_SECONDS})")
    p.add_argument("--frame-gap",     "-k", type=int,   default=FRAME_GAP,
                   help=f"Frame gap K for motion mask (default {FRAME_GAP})")
    p.add_argument("--speed-frames",  "-s", type=int,   default=SPEED_MIN_FRAMES,
                   help=f"Frames to cross all zones = speeding (default {SPEED_MIN_FRAMES})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    SPEED_MIN_FRAMES = args.speed_frames
    process_video(
        input_path   = args.input,
        output_path  = args.output,
        show_preview = args.preview,
        credits_sec  = args.credits,
        frame_gap    = args.frame_gap,
    )