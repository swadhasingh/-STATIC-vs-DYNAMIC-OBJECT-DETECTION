"""
Real-Time Traffic Analysis Pipeline
=====================================
Merges main.py + car_counter.py into ONE unified pipeline.

SOURCES:
  --source 0              Webcam/USB camera (acts as CCTV)
  --source video.mp4      Local video file
  --source rtsp://...     RTSP IP camera
  --source https://youtu.be/ID  YouTube (needs: pip install yt-dlp)

PIPELINE STEPS (single frame loop - no duplication):
  A  Read frame from any source
  B  Frame subtraction -> motion mask (frame N vs N-K)
  C  YOLOv8n object detection
  D  IOU Tracker -> persistent Track IDs
  E  Static/Dynamic classification:
       90pct rule: >90pct bbox pixels changed = DYNAMIC
       BBox centre drift: <14px over history = STATIC
       Infrastructure override: signs/lights always STATIC
  F  Car counter (centre-of-mass crossing horizontal line):
       Lookback method -- robust, no double-count
       Lane 1 (going away, UP) vs Lane 2 (coming close, DOWN)
  G  Speed zone detection (4 zones, flags SPEEDING)
  H  Annotated overlay + HUD drawn on frame
  I  Optional save to output video file + CSV crossing log

OVERLAY:
  Green box  = Dynamic vehicle/moving object
  Red box    = Static object
  Orange box = Speeding vehicle
  White dot  = Centre of mass + motion trail
  Cyan line  = Counting line (cars counted when crossing)
  Cyan dashes= Vertical lane boundary
  Yellow     = Speed zone lines

CONTROLS: Q=Quit  S=Snapshot  P=Pause/Resume

USAGE:
  python pipeline.py --source 0
  python pipeline.py --source traffic.mp4 --save
  python pipeline.py --source https://youtu.be/VIDEO_ID --save
  python pipeline.py --source rtsp://192.168.1.100/stream
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import csv
import subprocess
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO


# =====================================================================
#  CONFIGURATION
# =====================================================================

YOLO_MODEL         = "yolov8n.pt"
YOLO_CONF          = 0.38
YOLO_IOU           = 0.45
FRAME_GAP          = 6
PIXEL_DIFF_THRESH  = 25
STATIC_MOVE_THRESH = 14
STATIC_PIXEL_RATIO = 0.10
TRACK_HISTORY      = 20
TRACK_MAX_DISAP    = 25
IOU_MATCH_THRESH   = 0.30
COUNT_LINE_RATIO   = 0.50
DIRECTION_LOOKBACK = 8
NUM_H_ZONES        = 4
SPEED_MIN_FRAMES   = 35
CREDITS_SECONDS    = 10

# Colours BGR
C_VEH  = (0,  255,   0)
C_STA  = (0,    0, 255)
C_SPD  = (0,  140, 255)
C_L1   = (0,  255,   0)
C_L2   = (0,  200, 255)
C_UNC  = (180,180, 180)
C_LINE = (0,  255, 255)
C_DIV  = (0,  255, 255)
C_ZON  = (255,255,   0)
C_DOT  = (255,255, 255)
C_TRL  = (100,100, 100)
FONT   = cv2.FONT_HERSHEY_SIMPLEX

VEH_CLS = {"car","truck","bus","motorcycle","bicycle","train","boat"}
STA_CLS = {"stop sign","traffic light","parking meter","bench","fire hydrant"}


# =====================================================================
#  SOURCE RESOLVER
# =====================================================================

def resolve_source(source_str):
    """Handle webcam int / local file / RTSP / YouTube."""
    s = str(source_str).strip()
    try:
        idx = int(s)
        print(f"  [SOURCE] Webcam device {idx}")
        return idx, "webcam", True
    except ValueError:
        pass
    if "youtube.com" in s or "youtu.be" in s:
        print("  [SOURCE] YouTube -- resolving via yt-dlp ...")
        try:
            r = subprocess.run(
                ["yt-dlp", "-f", "best[ext=mp4]/best", "-g", s],
                capture_output=True, text=True, timeout=30)
            url = r.stdout.strip().split("\n")[0]
            if not url:
                raise RuntimeError("yt-dlp returned empty URL")
            print("  [SOURCE] URL resolved OK.")
            return url, "youtube", True
        except FileNotFoundError:
            print("  ERROR: yt-dlp not found. Run: pip install yt-dlp")
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR: {e}"); sys.exit(1)
    if any(s.startswith(p) for p in ("rtsp://", "http://", "https://")):
        print("  [SOURCE] Network stream")
        return s, "rtsp", True
    if os.path.isfile(s):
        print(f"  [SOURCE] Local file: {s}")
        return s, "file", False
    print(f"  ERROR: Cannot resolve source: {s}"); sys.exit(1)


# =====================================================================
#  UNIFIED TRACKER
#  Handles both static/dynamic classification AND counting line
# =====================================================================

class UnifiedTracker:

    def __init__(self):
        self.next_id     = 0
        self.objects     = {}
        self.disappeared = {}
        self.centre_hist = {}
        self.zone_hist   = {}
        self.lane_logged = {}
        self.counted     = {}

    @staticmethod
    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter==0: return 0.0
        return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

    def _register(self, det):
        tid = self.next_id
        cx,cy = (det[0]+det[2])//2,(det[1]+det[3])//2
        self.objects[tid]     = {"x1":det[0],"y1":det[1],"x2":det[2],"y2":det[3],
                                  "cx":cx,"cy":cy,"cls":det[4]}
        self.disappeared[tid] = 0
        self.centre_hist[tid] = deque(maxlen=TRACK_HISTORY)
        self.zone_hist[tid]   = deque(maxlen=SPEED_MIN_FRAMES+10)
        self.lane_logged[tid] = set()
        self.counted[tid]     = set()
        self.centre_hist[tid].append((cx,cy))
        self.next_id += 1
        return tid

    def _remove(self, tid):
        for d in [self.objects,self.disappeared,self.centre_hist,
                  self.zone_hist,self.lane_logged,self.counted]:
            d.pop(tid,None)

    def update(self, detections):
        if not detections:
            for tid in list(self.disappeared):
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
            return []
        if not self.objects:
            tids=[self._register(d) for d in detections]
            return [(*d,t) for d,t in zip(detections,tids)]
        eids   = list(self.objects.keys())
        eboxes = [[self.objects[i]["x1"],self.objects[i]["y1"],
                   self.objects[i]["x2"],self.objects[i]["y2"]] for i in eids]
        matched_d, matched_e = {}, set()
        for di,det in enumerate(detections):
            best,btid,bei = IOU_MATCH_THRESH,None,None
            for ei,tid in enumerate(eids):
                if ei in matched_e: continue
                s=self._iou(det[:4],eboxes[ei])
                if s>best: best,btid,bei=s,tid,ei
            if btid is not None:
                matched_d[di]=btid; matched_e.add(bei)
        for di,tid in matched_d.items():
            det=detections[di]
            cx,cy=(det[0]+det[2])//2,(det[1]+det[3])//2
            self.objects[tid].update({"x1":det[0],"y1":det[1],"x2":det[2],
                                       "y2":det[3],"cx":cx,"cy":cy,"cls":det[4]})
            self.centre_hist[tid].append((cx,cy))
            self.disappeared[tid]=0
        for di,det in enumerate(detections):
            if di not in matched_d:
                tid=self._register(det); matched_d[di]=tid
        for ei,tid in enumerate(eids):
            if ei not in matched_e:
                self.disappeared[tid]+=1
                if self.disappeared[tid]>TRACK_MAX_DISAP: self._remove(tid)
        return [(*detections[di],matched_d[di]) for di in range(len(detections))]

    def max_centre_drift(self, tid):
        h=self.centre_hist.get(tid)
        if not h or len(h)<2: return 999.0
        fx,fy=h[0]
        return max(((cx-fx)**2+(cy-fy)**2)**0.5 for cx,cy in h)

    def is_bbox_static(self, tid):
        return self.max_centre_drift(tid) < STATIC_MOVE_THRESH

    def update_zone(self, tid, zone):
        if tid in self.zone_hist: self.zone_hist[tid].append(zone)

    def is_speeding(self, tid):
        zh=list(self.zone_hist.get(tid,[]))
        if len(zh)<4: return False
        try:
            last_z4  = len(zh)-1-zh[::-1].index(4)
            first_z1 = next((i for i,z in enumerate(zh) if z==1 and i>last_z4),None)
            if first_z1 is not None and (first_z1-last_z4)<SPEED_MIN_FRAMES:
                return True
        except ValueError: pass
        return False

    def get_origin_cy(self, tid):
        h=self.centre_hist.get(tid)
        if not h: return None
        return h[max(0,len(h)-1-DIRECTION_LOOKBACK)][1]

    def already_counted(self, tid, direction):
        return direction in self.counted.get(tid,set())

    def mark_counted(self, tid, direction):
        if tid in self.counted: self.counted[tid].add(direction)


# =====================================================================
#  VISION HELPERS
# =====================================================================

def compute_motion_mask(g1, g2):
    """Step B: binary motion mask via frame difference."""
    diff=cv2.absdiff(g1,g2)
    blur=cv2.GaussianBlur(diff,(5,5),0)
    _,mask=cv2.threshold(blur,PIXEL_DIFF_THRESH,255,cv2.THRESH_BINARY)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    return mask


def pixel_ratio(mask,x1,y1,x2,y2):
    """Step E: fraction of bbox pixels active in motion mask (90pct rule)."""
    mh,mw=mask.shape
    x1,y1=max(0,x1),max(0,y1); x2,y2=min(mw,x2),min(mh,y2)
    if x2<=x1 or y2<=y1: return 0.0
    roi=mask[y1:y2,x1:x2]
    return cv2.countNonZero(roi)/roi.size if roi.size>0 else 0.0


def h_zone(cy,H):
    """Step G: horizontal speed zone 1-4."""
    return min(int(cy//(H/NUM_H_ZONES))+1, NUM_H_ZONES)


def draw_scene_overlay(frame,W,H,line_y):
    """Step H: draw counting line, lane divider, speed zone dashes."""
    # Counting line (solid cyan)
    cv2.line(frame,(0,line_y),(W,line_y),C_LINE,2)
    cv2.putText(frame,"COUNTING LINE",(W//2-65,line_y-8),
                FONT,0.48,C_LINE,1,cv2.LINE_AA)
    # Vertical lane divider (dashed cyan)
    for y in range(0,H,22):
        cv2.line(frame,(W//2,y),(W//2,min(y+11,H)),C_DIV,1)
    # Horizontal speed zones (dashed yellow)
    zh=H//NUM_H_ZONES
    for i in range(1,NUM_H_ZONES):
        yy=i*zh
        for x in range(0,W,28):
            cv2.line(frame,(x,yy),(min(x+14,W),yy),C_ZON,1)
    # Lane labels bottom
    cv2.putText(frame,"L1: Going Away",(8,H-12),FONT,0.48,C_L1,1,cv2.LINE_AA)
    cv2.putText(frame,"L2: Coming Close",(W//2+6,H-12),FONT,0.48,C_L2,1,cv2.LINE_AA)


def draw_hud(frame,W,H,st,fps_now,src_type,paused):
    """Step H: HUD overlay with counts and source info."""
    cv2.putText(frame,f"SOURCE: {src_type.upper()}",(8,20),
                FONT,0.50,(200,200,200),1,cv2.LINE_AA)
    mode="** PAUSED **" if paused else f"LIVE  {fps_now:.1f} fps"
    cv2.putText(frame,mode,(8,40),FONT,0.50,
                (0,0,255) if paused else (0,255,100),1,cv2.LINE_AA)
    rows=[
        (f"Lane 1 (Away)  :{st['l1']:>5}", C_L1),
        (f"Lane 2 (Close) :{st['l2']:>5}", C_L2),
        (f"TOTAL COUNTED  :{st['tot']:>5}", C_LINE),
        (f"Dynamic objs   :{st['dyn']:>5}", C_VEH),
        (f"Static objs    :{st['sta']:>5}", C_STA),
        (f"Speeding       :{st['spd']:>5}", C_SPD),
        (f"Frame          :{st['frm']:>5}", (180,180,180)),
    ]
    for i,(txt,col) in enumerate(rows):
        tw=cv2.getTextSize(txt,FONT,0.52,1)[0][0]
        cv2.putText(frame,txt,(W-tw-8,20+i*22),FONT,0.52,col,1,cv2.LINE_AA)
    cv2.putText(frame,"Q=Quit  S=Snapshot  P=Pause",
                (8,H-30),FONT,0.40,(150,150,150),1,cv2.LINE_AA)


# =====================================================================
#  MAIN PIPELINE
# =====================================================================

def run_pipeline(source, save=False, output_path="pipeline_output.mp4",
                 log_path="pipeline_log.csv", show_window=True,
                 credits_sec=CREDITS_SECONDS, frame_gap=FRAME_GAP):

    # Step A: resolve source
    cv2_src, src_type, is_live = resolve_source(source)

    cap=cv2.VideoCapture(cv2_src)
    if is_live: cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {cv2_src}"); sys.exit(1)

    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W==0 or H==0:
        ret,probe=cap.read()
        if ret: H,W=probe.shape[:2]; cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        else: print("  ERROR: Cannot read frame."); sys.exit(1)

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
    credits_frames = int(fps*credits_sec) if (not is_live and total_frames>0) else 0
    cropped_frames = max(total_frames-credits_frames,0) if total_frames>0 else -1
    LINE_Y         = int(H*COUNT_LINE_RATIO)

    print(f"\n{'='*70}")
    print(f"  REAL-TIME TRAFFIC ANALYSIS PIPELINE  --  NMIMS")
    print(f"{'='*70}")
    print(f"  Source        : {src_type.upper()}")
    print(f"  Resolution    : {W}x{H}  |  FPS: {fps:.1f}")
    if not is_live and total_frames>0:
        print(f"  Duration      : {total_frames/fps:.1f}s  ({total_frames:,} frames)")
        print(f"  Credits trim  : last {credits_sec}s removed")
    else:
        print(f"  Mode          : LIVE STREAM  (no frame limit)")
    print(f"  Counting line : Y={LINE_Y}px  ({COUNT_LINE_RATIO*100:.0f}% of height)")
    print(f"  YOLO model    : {YOLO_MODEL}  |  conf={YOLO_CONF}")
    print(f"  Frame gap K   : {frame_gap}  |  Lookback: {DIRECTION_LOOKBACK} frames")
    print(f"  Save output   : {save}  ->  {output_path if save else 'N/A'}")
    print(f"{'='*70}")
    print(f"\n  Controls:  Q=Quit   S=Snapshot   P=Pause/Resume\n")

    writer=None
    if save:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".",exist_ok=True)
        writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))

    print("  Loading YOLO model ...")
    model=YOLO(YOLO_MODEL)
    tracker=UnifiedTracker()
    print("  YOLO ready.  Pipeline running ...\n")

    # Counters and state
    lane1=0; lane2=0; total=0
    spd_ids=set(); cls_cnt=defaultdict(int)
    dyn_n=0; sta_n=0; log=[]; snaps=0
    flashes={}; fbuf=[]; fidx=0; paused=False; t0=time.time()

    # ================================================================
    #  FRAME LOOP
    # ================================================================
    while True:

        if paused:
            key=cv2.waitKey(50)&0xFF
            if key==ord("q"): break
            if key==ord("p"): paused=False; print("  [RESUMED]")
            continue

        # Step A: read frame
        ret,frame=cap.read()
        if not ret:
            if is_live:
                print("  [WARN] Read failed -- retrying ...")
                time.sleep(0.5)
                ret,frame=cap.read()
                if not ret: print("  Stream ended."); break
            else:
                print("  End of file."); break

        if not is_live and cropped_frames>0 and fidx>=cropped_frames:
            print(f"\n  [CUT] Credits at frame {fidx}."); break

        # Step B: motion mask
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        fbuf.append(gray)
        if len(fbuf)>frame_gap+1: fbuf.pop(0)
        mask=compute_motion_mask(fbuf[0],fbuf[-1]) if len(fbuf)==frame_gap+1              else np.zeros((H,W),dtype=np.uint8)

        annotated=frame.copy()
        draw_scene_overlay(annotated,W,H,LINE_Y)

        # Step C: YOLO detection
        results=model.predict(source=frame,conf=YOLO_CONF,iou=YOLO_IOU,verbose=False)
        raw=[]
        if results and results[0].boxes is not None:
            boxes=results[0].boxes
            class_ids=boxes.cls.cpu().numpy().astype(int)
            coords=boxes.xyxy.cpu().numpy().astype(int)
            names=model.names
            for box,cid in zip(coords,class_ids):
                raw.append((*box.tolist(),names[cid].lower()))

        # Step D: track
        tracked=tracker.update(raw)

        # Steps E + F + G: per-object logic
        for det in tracked:
            x1,y1,x2,y2,cls,tid=det
            cx,cy=(x1+x2)//2,(y1+y2)//2
            ratio=pixel_ratio(mask,x1,y1,x2,y2)
            is_veh=cls in VEH_CLS
            is_infra=cls in STA_CLS

            # Step E: static / dynamic classify
            if is_veh:
                is_dyn=True; is_spd=False
                zone=h_zone(cy,H)
                tracker.update_zone(tid,zone)
                if tracker.is_speeding(tid): spd_ids.add(tid); is_spd=True
                color=C_SPD if is_spd else C_VEH
                lbl=f"{cls}{'_SPD' if is_spd else ''} #{tid}"
            elif is_infra:
                is_dyn=False; color=C_STA; lbl=f"{cls}|STA #{tid}"
            else:
                if   ratio>0.90:               is_dyn=True
                elif ratio<STATIC_PIXEL_RATIO: is_dyn=False
                else:                          is_dyn=not tracker.is_bbox_static(tid)
                color=C_VEH if is_dyn else C_STA
                lbl=f"{cls}|{'DYN' if is_dyn else 'STA'} #{tid}"

            if is_dyn: dyn_n+=1
            else:      sta_n+=1
            cls_cnt[cls]+=1

            # Step F: counting line crossing (lookback method from car_counter.py)
            orig_cy=tracker.get_origin_cy(tid)
            direction=None
            if orig_cy is not None and len(tracker.centre_hist.get(tid,[]))>3:
                if orig_cy>LINE_Y and cy<LINE_Y and not tracker.already_counted(tid,"up"):
                    lane1+=1; total+=1
                    tracker.mark_counted(tid,"up"); direction="up"
                    flashes[tid]=8
                    print(f"  UP L1 | f{fidx:>6} id{tid:>4} {cls:<12} | L1={lane1} Tot={total}")
                elif orig_cy<LINE_Y and cy>LINE_Y and not tracker.already_counted(tid,"down"):
                    lane2+=1; total+=1
                    tracker.mark_counted(tid,"down"); direction="down"
                    flashes[tid]=8
                    print(f"  DN L2 | f{fidx:>6} id{tid:>4} {cls:<12} | L2={lane2} Tot={total}")

            if direction:
                log.append({"frame":fidx,"tid":tid,"class":cls,
                            "dir":direction,"cx":cx,"cy":cy,
                            "l1":lane1,"l2":lane2,"total":total,
                            "time":datetime.now().strftime("%H:%M:%S")})

            if flashes.get(tid,0)>0:
                cv2.line(annotated,(0,LINE_Y),(W,LINE_Y),(255,255,255),5)
                flashes[tid]-=1

            # Step H: draw thin box + label (no filled band)
            if is_veh:
                if   "down" in tracker.counted.get(tid,set()): dc=C_L2
                elif "up"   in tracker.counted.get(tid,set()): dc=C_L1
                else:                                          dc=C_UNC
            else:
                dc=color

            cv2.rectangle(annotated,(x1,y1),(x2,y2),dc,1)
            cv2.putText(annotated,lbl,(x1+2,max(y1-5,10)),FONT,0.36,dc,1,cv2.LINE_AA)
            cv2.circle(annotated,(cx,cy),4,C_DOT,-1)
            trail=list(tracker.centre_hist.get(tid,[]))
            for i in range(1,len(trail)):
                cv2.line(annotated,trail[i-1],trail[i],C_TRL,1)

        # Step H: draw HUD
        elapsed=time.time()-t0
        fps_now=fidx/elapsed if elapsed>0 else 0.0
        draw_hud(annotated,W,H,
                 {"l1":lane1,"l2":lane2,"tot":total,
                  "dyn":dyn_n,"sta":sta_n,"spd":len(spd_ids),"frm":fidx},
                 fps_now,src_type,paused)

        # Step I: save frame
        if writer: writer.write(annotated)

        if show_window:
            cv2.imshow("Traffic Analysis Pipeline  [Q=quit  S=snap  P=pause]",annotated)
            key=cv2.waitKey(1)&0xFF
            if   key==ord("q"): print("\n  [QUIT]"); break
            elif key==ord("p"): paused=True; print("  [PAUSED] Press P to resume.")
            elif key==ord("s"):
                sn=f"snapshot_{snaps:03d}.jpg"
                cv2.imwrite(sn,annotated)
                print(f"  [SNAP] Saved: {sn}"); snaps+=1

        fidx+=1
        if fidx%200==0:
            el=time.time()-t0
            print(f"  [{fidx:>6}f | {fidx/el:.1f}fps] "
                  f"L1={lane1} L2={lane2} Tot={total} | "
                  f"Dyn={dyn_n} Sta={sta_n} Spd={len(spd_ids)}")

    # Cleanup
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    if log:
        with open(log_path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["frame","tid","class","dir","cx","cy",
                                            "l1","l2","total","time"])
            w.writeheader(); w.writerows(log)
        print(f"\n  Crossings log saved: {log_path}")

    total_time=time.time()-t0
    sep="="*70
    print(f"\n{sep}")
    print(f"{'  PIPELINE FINAL REPORT':^70}")
    print(f"{sep}")
    print(f"  Source         : {src_type.upper()}")
    print(f"  Frames proc.   : {fidx:,}")
    print(f"  Run time       : {total_time:.1f}s  ({fidx/total_time:.1f} fps avg)")
    print()
    print(f"  CAR COUNT  (centre-of-mass line crossing method)")
    print(f"  {'Lane 1 -- Going Away    (UP)  ':<42} {lane1:>6,}")
    print(f"  {'Lane 2 -- Coming Close  (DOWN)':<42} {lane2:>6,}")
    print(f"  {'─'*42}   {'─'*6}")
    print(f"  {'TOTAL VEHICLES COUNTED':<42} {total:>6,}")
    print()
    print(f"  OBJECT CLASSIFICATION")
    print(f"  {'Dynamic (moving) detections':<42} {dyn_n:>6,}")
    print(f"  {'Static  (stopped) detections':<42} {sta_n:>6,}")
    print(f"  {'Speeding vehicles (unique IDs)':<42} {len(spd_ids):>6,}")
    print()
    print(f"  PER-CLASS DETECTIONS")
    print(f"  {'CLASS':<22} {'COUNT':>8}")
    print(f"  {'─'*22}   {'─'*8}")
    for cls,cnt in sorted(cls_cnt.items(),key=lambda x:-x[1]):
        mk=">>" if cls in VEH_CLS else "  "
        print(f"  {mk} {cls:<20} {cnt:>8,}")
    print(f"  (>> = vehicle class)")
    if save: print(f"\n  Output video : {output_path}")
    print(f"{sep}\n")



if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="Real-Time Traffic Analysis Pipeline")
    p.add_argument("--source","-s",required=True)
    p.add_argument("--save","-w",action="store_true")
    p.add_argument("--output","-o",default="pipeline_output.mp4")
    p.add_argument("--log","-l",default="pipeline_log.csv")
    p.add_argument("--no-window",action="store_true")
    p.add_argument("--credits","-c",type=float,default=CREDITS_SECONDS)
    p.add_argument("--frame-gap","-k",type=int,default=FRAME_GAP)
    p.add_argument("--line","-y",type=float,default=COUNT_LINE_RATIO)
    args=p.parse_args()
    COUNT_LINE_RATIO=args.line
    run_pipeline(args.source,args.save,args.output,args.log,
                 not args.no_window,args.credits,args.frame_gap)