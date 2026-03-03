"""
=============================================================
  CAR COUNTER v2 - CENTRE OF MASS + COUNTING LINE METHOD
  FIXED: Crossing detection now works correctly
  - Uses direction buffer (not single-frame jump)
  - Counts car the moment its centre passes the line
  - Works even when cars start near or on the line
=============================================================
"""
import cv2
import numpy as np
import csv
import argparse
import time
from collections import deque, defaultdict
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"
YOLO_CONF        = 0.35          # slightly lower conf to catch more vehicles
YOLO_IOU         = 0.45
CREDITS_SECONDS  = 10
COUNT_LINE_RATIO = 0.50          # counting line at 50% frame height
TRACK_MAX_DISAP  = 30
IOU_MATCH_THRESH = 0.30          # lower = more lenient matching
CENTRE_HIST      = 40

# How many frames to look BACK to determine "where the car came from"
# This avoids missing cars that were already near the line at start
DIRECTION_LOOKBACK = 8

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
FONT            = cv2.FONT_HERSHEY_SIMPLEX
COLOR_LINE      = (0, 255, 255)   # cyan counting line
COLOR_LANE1     = (0, 255,   0)   # green  - Lane 1 going away (UP)
COLOR_LANE2     = (0, 140, 255)   # orange - Lane 2 coming close (DOWN)
COLOR_UNCOUNTED = (180,180,180)   # grey   - not yet crossed
COLOR_CENTRE    = (255,255,255)   # white dot


class VehicleTracker:
    def __init__(self):
        self.next_id=0; self.objects={}; self.disappeared={}
        self.centres={}; self.counted={}

    def _iou(self,a,b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter==0: return 0.0
        return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

    def _register(self,det):
        tid=self.next_id
        cx,cy=(det[0]+det[2])//2,(det[1]+det[3])//2
        self.objects[tid]=det; self.disappeared[tid]=0
        self.centres[tid]=deque(maxlen=CENTRE_HIST)
        self.centres[tid].append((cx,cy))
        self.counted[tid]=set(); self.next_id+=1; return tid

    def _remove(self,tid):
        for d in [self.objects,self.disappeared,self.centres,self.counted]:
            d.pop(tid,None)

    def update(self,detections):
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
            return []
        if not self.objects:
            return [(*d,self._register(d)) for d in detections]
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
            self.objects[tid]=det; self.disappeared[tid]=0
            self.centres[tid].append((cx,cy))
        for di,det in enumerate(detections):
            if di not in matched_d:
                tid=self._register(det); matched_d[di]=tid
        for ei,tid in enumerate(eids):
            if ei not in matched_e:
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
        return [(*detections[di],matched_d[di]) for di in range(len(detections))]

    def get_centre(self,tid):
        h=self.centres.get(tid)
        return h[-1] if h else (None,None)

    def get_origin_cy(self,tid):
        """Return the cy from DIRECTION_LOOKBACK frames ago (or earliest available)."""
        h=self.centres.get(tid)
        if not h: return None
        idx=max(0,len(h)-1-DIRECTION_LOOKBACK)
        return h[idx][1]

    def mark_counted(self,tid,direction):
        if tid in self.counted: self.counted[tid].add(direction)

    def already_counted(self,tid,direction):
        return direction in self.counted.get(tid,set())


def count_cars(input_path, output_path="car_count_output.mp4",
               log_path="car_count_log.csv", show_preview=False,
               credits_sec=CREDITS_SECONDS):

    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {input_path}")
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    credits_frames=int(fps*credits_sec)
    cropped_frames=max(total_frames-credits_frames,0)
    LINE_Y=int(H*COUNT_LINE_RATIO)

    print(f"\n{'='*65}")
    print(f"  CAR COUNTER v2 - Centre of Mass Crossing Line (FIXED)")
    print(f"{'='*65}")
    print(f"  Input         : {input_path}")
    print(f"  Resolution    : {W}x{H}  |  FPS: {fps:.2f}")
    print(f"  Total frames  : {total_frames:,}  |  Cropped: {cropped_frames:,}")
    print(f"  Counting line : Y={LINE_Y}px ({COUNT_LINE_RATIO*100:.0f}% height)")
    print(f"  Direction buf : {DIRECTION_LOOKBACK} frames lookback")
    print(f"  Lane 1 (Away) : cars moving UP   (centre was below, now above line)")
    print(f"  Lane 2 (Close): cars moving DOWN (centre was above, now below line)")
    print(f"{'='*65}\n")

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out_vid=cv2.VideoWriter(output_path,fourcc,fps,(W,H))
    model=YOLO(YOLO_MODEL)
    tracker=VehicleTracker()
    print("  YOLO loaded. Counting...\n")

    lane1_count=0; lane2_count=0; total_count=0
    class_counts=defaultdict(lambda:{"lane1":0,"lane2":0})
    crossing_log=[]; frame_idx=0; t_start=time.time()
    flash_frames={}  # tid -> frames left to flash

    while True:
        ret,frame=cap.read()
        if not ret: break
        if frame_idx>=cropped_frames:
            print(f"\n  [CUT] Credits at frame {frame_idx}."); break

        annotated=frame.copy()

        # ── Draw counting line ────────────────────────────────────────────
        cv2.line(annotated,(0,LINE_Y),(W,LINE_Y),COLOR_LINE,2)
        cv2.putText(annotated,"COUNTING LINE",
                    (W//2-70,LINE_Y-10),FONT,0.55,COLOR_LINE,2,cv2.LINE_AA)

        # Vertical lane divider (dashed yellow)
        for y in range(0,H,22):
            cv2.line(annotated,(W//2,y),(W//2,min(y+11,H)),(255,255,0),1)

        cv2.putText(annotated,"LANE 1: Going Away (UP)",
                    (10,H-14),FONT,0.55,COLOR_LANE1,1,cv2.LINE_AA)
        cv2.putText(annotated,"LANE 2: Coming Close (DOWN)",
                    (W//2+8,H-14),FONT,0.55,COLOR_LANE2,1,cv2.LINE_AA)

        # ── YOLO ─────────────────────────────────────────────────────────
        results=model.predict(source=frame,conf=YOLO_CONF,iou=YOLO_IOU,verbose=False)
        raw=[]
        if results and results[0].boxes is not None:
            boxes=results[0].boxes
            class_ids=boxes.cls.cpu().numpy().astype(int)
            coords=boxes.xyxy.cpu().numpy().astype(int)
            names=model.names
            for box,cid in zip(coords,class_ids):
                cls_name=names[cid].lower()
                if cls_name in VEHICLE_CLASSES:
                    raw.append([*box.tolist(),cls_name])

        tracked=tracker.update(raw)

        for det in tracked:
            x1,y1,x2,y2,cls_name,tid=det
            cx,cy=tracker.get_centre(tid)
            if cx is None: continue

            # ── CROSSING DETECTION (lookback method) ──────────────────────
            # Get where this car's centre WAS several frames ago
            origin_cy=tracker.get_origin_cy(tid)

            direction=None

            if origin_cy is not None and len(tracker.centres.get(tid,[]))>3:
                # Car was BELOW line before, now ABOVE = going UP = Lane 1
                if (origin_cy > LINE_Y) and (cy < LINE_Y) and \
                   not tracker.already_counted(tid,"up"):
                    lane1_count+=1; total_count+=1
                    class_counts[cls_name]["lane1"]+=1
                    tracker.mark_counted(tid,"up"); direction="up"
                    flash_frames[tid]=8
                    print(f"  UP  Lane1 | Frame {frame_idx:>6} | ID {tid:>4} | "
                          f"{cls_name:<12} | L1={lane1_count}  Total={total_count}")

                # Car was ABOVE line before, now BELOW = going DOWN = Lane 2
                elif (origin_cy < LINE_Y) and (cy > LINE_Y) and \
                     not tracker.already_counted(tid,"down"):
                    lane2_count+=1; total_count+=1
                    class_counts[cls_name]["lane2"]+=1
                    tracker.mark_counted(tid,"down"); direction="down"
                    flash_frames[tid]=8
                    print(f"  DN  Lane2 | Frame {frame_idx:>6} | ID {tid:>4} | "
                          f"{cls_name:<12} | L2={lane2_count}  Total={total_count}")

            if direction:
                crossing_log.append({"frame":frame_idx,"tid":tid,"class":cls_name,
                    "direction":direction,"cx":cx,"cy":cy,
                    "lane1_running":lane1_count,"lane2_running":lane2_count,
                    "total_running":total_count})

            # Flash line white on crossing
            if flash_frames.get(tid,0)>0:
                cv2.line(annotated,(0,LINE_Y),(W,LINE_Y),(255,255,255),5)
                flash_frames[tid]-=1

            # Box colour by counted state
            if "down" in tracker.counted.get(tid,set()):   box_col=COLOR_LANE2
            elif "up" in tracker.counted.get(tid,set()):   box_col=COLOR_LANE1
            else:                                           box_col=COLOR_UNCOUNTED

            # Thin bbox, no filled band
            cv2.rectangle(annotated,(x1,y1),(x2,y2),box_col,1)
            cv2.putText(annotated,f"{cls_name}#{tid}",
                        (x1+2,max(y1-5,10)),FONT,0.36,box_col,1,cv2.LINE_AA)

            # Centre of mass dot
            cv2.circle(annotated,(cx,cy),5,COLOR_CENTRE,-1)

            # Motion trail
            trail=list(tracker.centres.get(tid,[]))
            for i in range(1,len(trail)):
                alpha=i/len(trail)
                c=int(180*alpha)
                cv2.line(annotated,trail[i-1],trail[i],(c,c,c),1)

        # ── HUD top-right ──────────────────────────────────────────────────
        hud=[
            (f"Lane 1 (Away) : {lane1_count:>5}",COLOR_LANE1),
            (f"Lane 2 (Close): {lane2_count:>5}",COLOR_LANE2),
            (f"TOTAL         : {total_count:>5}",(0,255,255)),
            (f"Frame {frame_idx:,}/{cropped_frames:,}",(180,180,180)),
        ]
        for li,(txt,col) in enumerate(hud):
            tw=cv2.getTextSize(txt,FONT,0.65,2)[0][0]
            cv2.putText(annotated,txt,(W-tw-12,30+li*28),FONT,0.65,col,2,cv2.LINE_AA)

        out_vid.write(annotated)
        if show_preview:
            cv2.imshow("Car Counter v2 [Q=quit]",annotated)
            if cv2.waitKey(1)&0xFF==ord("q"):
                print("\n  Preview closed."); break
        frame_idx+=1

    cap.release(); out_vid.release()
    if show_preview: cv2.destroyAllWindows()

    with open(log_path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["frame","tid","class","direction",
            "cx","cy","lane1_running","lane2_running","total_running"])
        writer.writeheader(); writer.writerows(crossing_log)

    elapsed=time.time()-t_start
    w=65
    print(f"\n{'='*w}")
    print(f"{'  CAR COUNT FINAL REPORT':^{w}}")
    print(f"{'='*w}")
    print(f"  Method    : Centre-of-Mass crossing horizontal counting line")
    print(f"  Line pos  : Y={LINE_Y}px ({COUNT_LINE_RATIO*100:.0f}% of frame height)")
    print(f"  Lookback  : {DIRECTION_LOOKBACK} frames to determine travel direction")
    print()
    print(f"  {'LANE':<35} {'COUNT':>8}")
    print(f"  {'─'*35}   {'─'*8}")
    print(f"  Lane 1 - Going AWAY    (UP)        {lane1_count:>8,}")
    print(f"  Lane 2 - Coming CLOSE  (DOWN)      {lane2_count:>8,}")
    print(f"  {'─'*35}   {'─'*8}")
    print(f"  TOTAL VEHICLES COUNTED             {total_count:>8,}")
    print()
    print(f"  PER-CLASS BREAKDOWN:")
    print(f"  {'CLASS':<16} {'LANE1 (UP)':>10} {'LANE2 (DOWN)':>12} {'TOTAL':>8}")
    print(f"  {'─'*16}   {'─'*10}   {'─'*12}   {'─'*8}")
    for cls in sorted(class_counts):
        l1=class_counts[cls]["lane1"]; l2=class_counts[cls]["lane2"]
        print(f"  {cls:<16} {l1:>10,} {l2:>12,} {l1+l2:>8,}")
    print()
    print(f"  Crossings CSV   : {log_path}")
    print(f"  Annotated video : {output_path}")
    print(f"  Time            : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
    print(f"{'='*w}\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description="Car Counter v2 - Fixed Crossing Detection")
    p.add_argument("--input", "-i",required=True)
    p.add_argument("--output","-o",default="car_count_output.mp4")
    p.add_argument("--log",   "-l",default="car_count_log.csv")
    p.add_argument("--preview","-p",action="store_true")
    p.add_argument("--credits","-c",type=float,default=CREDITS_SECONDS)
    p.add_argument("--line",  "-y",type=float,default=COUNT_LINE_RATIO,
                   help="Counting line as fraction of height (default 0.5)")
    args=p.parse_args()
    COUNT_LINE_RATIO=args.line
    count_cars(args.input,args.output,args.log,args.preview,args.credits)