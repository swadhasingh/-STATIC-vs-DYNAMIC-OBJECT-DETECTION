# Static vs Dynamic Object Detection & Traffic Analysis Pipeline


> Real-time object classification, vehicle counting, and live stream analysis using YOLOv8 + custom tracking.

---

## What This Project Does

This pipeline takes any video source — a local file, a YouTube stream, an IP camera, or a webcam — and performs **real-time**:

1. **Static vs Dynamic object detection** using a dual-method classifier
2. **Vehicle counting** via centre-of-mass line-crossing
3. **Speed zone detection** across 4 horizontal zones
4. **Classroom people counting** with live occupancy tracking

All in a single unified frame loop with no duplication between steps.

---

## Demo

| Traffic Pipeline | Classroom Monitor |
|---|---|
| Green box = Dynamic vehicle | Green box = Person entering |
| Red box = Static object | Red box = Person leaving |
| Orange box = Speeding | Blue box = Static furniture |
| Cyan line = Counting line | Big number = Live occupancy |

---

## Project Structure

```
.
├── pipeline.py                # Main unified pipeline (traffic / YouTube / RTSP)
├── main.py                    # Steps 1-7 combined (static/dynamic detection core)
├── car_counter.py             # Standalone car counter (centre-of-mass method)
├── classroom_pipeline.py      # Webcam classroom people counter + occupancy
├── test1_dynamic_cars.py      # Confirmatory Test 1: dynamic car detection
├── test2_static_elements.py   # Confirmatory Test 2: static element detection
├── test3_90pct_rule.py        # Confirmatory Test 3: 90% threshold rule validation
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Pipeline Steps (Steps 1–7)

| Step | Description |
|------|-------------|
| 1 | Load video from any source (file / webcam / stream) |
| 2 | Crop end-credits from local videos |
| 3 | YOLOv8n real-time object detection overlay |
| 4 | Count objects per frame |
| 5 | Frame subtraction → motion mask (frame N vs N−K) |
| 6 | Union of frame-diff and bounding box → static or dynamic label |
| 7 | % pixel overlap (90% rule) threshold decision |

### The 90% Rule (Step 7)
```
pixel_ratio = active motion pixels inside bbox / total bbox pixels

ratio > 90%  →  DEFINITELY DYNAMIC  (object clearly moved)
ratio < 10%  →  DEFINITELY STATIC   (object barely moved)
10% – 90%    →  TIEBREAK via bbox centre tracking
```
This correctly handles **slowing/decelerating vehicles** that would otherwise be wrongly classified as static.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/static-vs-dynamic-object-detection.git
cd static-vs-dynamic-object-detection

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> The YOLOv8 model (`yolov8n.pt`) downloads automatically on first run (~6MB).

---

## Usage

### Traffic / YouTube / RTSP Pipeline
```bash
# Local video file
python pipeline.py --source traffic.mp4 --save

# Webcam as CCTV
python pipeline.py --source 0

# YouTube live stream
python pipeline.py --source "https://www.youtube.com/watch?v=VIDEO_ID" --save

# RTSP IP camera
python pipeline.py --source "rtsp://192.168.1.100/stream" --save

# Save output + headless mode (no window)
python pipeline.py --source 0 --save --output out.mp4 --no-window
```

### Classroom People Counter
```bash
# Webcam
python classroom_pipeline.py --source 0

# Save recording
python classroom_pipeline.py --source 0 --save --output classroom_out.mp4
```

### Standalone Car Counter
```bash
python car_counter.py --source traffic.mp4 --save
```

### Confirmatory Tests
```bash
python test1_dynamic_cars.py -i traffic.mp4 -p
python test2_static_elements.py -i traffic.mp4
python test3_90pct_rule.py -i traffic.mp4
```

---

## Keyboard Controls (live window)

| Key | Action |
|-----|--------|
| `Q` | Quit and print final report |
| `S` | Save snapshot image |
| `P` | Pause / Resume |

---

## Confirmatory Tests

Three independent test scripts validate the core logic:

| Test | Purpose |
|------|---------|
| **Test 1** — `test1_dynamic_cars.py` | Confirms all vehicles show white motion spots (dynamic) in frame difference. Side-by-side output: YOLO boxes + motion mask. |
| **Test 2** — `test2_static_elements.py` | Detects all objects whose centre-of-mass moves less than a pixel threshold. RED = static confirmed. |
| **Test 3** — `test3_90pct_rule.py` | Validates the 90% rule prevents slowing vehicles from being wrongly classified as static. **Orange border** = slowing vehicle correctly saved as DYNAMIC. |

---

## Static vs Dynamic Classification

```
For VEHICLES:
  Always classified as DYNAMIC (green)
  Exception: speeding check via zone history

For INFRASTRUCTURE (signs, traffic lights):
  Always classified as STATIC (overridden, not pixel-checked)

For ALL OTHER objects:
  Step 1 — pixel_ratio > 90%  →  DYNAMIC
  Step 2 — pixel_ratio < 10%  →  STATIC
  Step 3 — in between         →  check bbox centre movement history
             centre drift < 14px over 15 frames  →  STATIC
             centre drift >= 14px                →  DYNAMIC
```

---

## Car Counting Method

Counting uses the **centre-of-mass line-crossing method**:

```
A horizontal counting line is drawn at 50% of frame height.

For each tracked vehicle:
  origin_cy = centre Y position from 8 frames ago (lookback)
  current_cy = centre Y position now

  origin_cy > LINE  AND  current_cy < LINE  →  crossed UP   →  Lane 1 (going away)
  origin_cy < LINE  AND  current_cy > LINE  →  crossed DOWN →  Lane 2 (coming close)

Each Track ID counted EXACTLY ONCE per direction.
Dead-zone prevents double counting.
```

The **lookback method** (vs single-frame jump detection) makes this robust to:
- Slow-moving vehicles
- Vehicles that start near the counting line
- Frame drops in live streams

---

## Live Results (YouTube Traffic Cam Test)

```
Source       : YOUTUBE
Frames proc. : 3,404  (~3.75 minutes)
Avg speed    : 15.1 fps

Lane 1 (Going Away)  :  36 vehicles
Lane 2 (Coming Close):  31 vehicles
TOTAL COUNTED        :  67 vehicles

Dynamic detections   : 27,038
Static detections    : 16,819
```

---

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy<2
yt-dlp              # for YouTube streams only
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Notes

- **`yolov8n.pt`** is not included in this repo (auto-downloads on first run)
- **Video files** are not included (use your own or a YouTube link)
- For YouTube support: `pip install yt-dlp`
- Tested on macOS with Apple Silicon (M-series) and Python 3.12

---

## Author

Swadha Singh
Edge Computing / Computer Vision Project