import cv2
import time
import numpy as np
from collections import defaultdict
import requests
from ultralytics import YOLO

# ── config ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
SHARPNESS_THRESHOLD  = 5.0
DEVICE               = 'mps'

# ── webapp endpoint ───────────────────────────────────────────────────────────
WEBAPP_URL = "http://localhost:8001/update-vision"

# ── camera source ─────────────────────────────────────────────────────────────
USE_USB      = True
DROIDCAM_URL = "http://10.165.99.206:4747/video"
CAMERA_INDEX = 1  # iPhone wired via USB

WARMUP_SECONDS    = 1.0
MAX_MISSED_FRAMES = 30
INFERENCE_WIDTH   = 640
INFERENCE_HEIGHT  = 480

# ── camera motion detection ───────────────────────────────────────────────────
CAM_MOTION_THRESHOLD    = 0.04   # fraction of frame dimension
CAM_MOTION_FREEZE_FRAMES = 12    # frames to stay frozen after motion stops

# ── detection filters ─────────────────────────────────────────────────────────
ALLOWED_CLASSES   = set(range(100))  # accept any class the model outputs
BLACKLIST_CLASSES = set()
MIN_BBOX_AREA     = 3000
MAX_BBOX_AREA     = 550000
MIN_ASPECT_RATIO  = 0.5

# ── boundary config ───────────────────────────────────────────────────────────
BOUNDARY_AXIS = 'x'
BOUNDARY_POS  = 0.5
INWARD_DIR    = 'right'

# ── zone tracker config ───────────────────────────────────────────────────────
ZONE_WIDTH        = 120
ADD_COOLDOWN_SEC  = 3.0
STATE_MEMORY_SEC  = 3.0
EVENT_COOLDOWN    = 45
MAX_EVENT_LOG     = 8
RELINK_RADIUS     = 120
RELINK_TIMEOUT    = 90

# ── model ─────────────────────────────────────────────────────────────────────
import os as _os
_VISION_DIR = _os.path.dirname(_os.path.abspath(__file__))
model = YOLO(_os.path.join(_VISION_DIR, 'trained_model.pt'))
model.to(DEVICE)
print("warming up MPS...")
model(np.zeros((INFERENCE_HEIGHT, INFERENCE_WIDTH, 3), dtype=np.uint8), verbose=False)
print("MPS ready")

# ── class label correction ────────────────────────────────────────────────────
# Confirmed mapping (raw model output → actual item):
#   raw lemon    → coke_can
#   raw coke_can → doritos
#   raw doritos  → lemon
CLASS_REMAP = {
    'lemon':    'coke_can',
    'coke_can': 'doritos',
    'doritos':  'lemon',
}

# ── boundary helpers ──────────────────────────────────────────────────────────
def _line() -> float:
    return (BOUNDARY_POS * INFERENCE_HEIGHT if BOUNDARY_AXIS == 'y'
            else BOUNDARY_POS * INFERENCE_WIDTH)

def _zone_near() -> float:
    return _line() - ZONE_WIDTH

def _zone_far() -> float:
    return _line() + ZONE_WIDTH

def _axis_val_leading(bbox) -> float:
    x1, y1, x2, y2 = bbox
    if BOUNDARY_AXIS == 'x':
        return x2 if INWARD_DIR == 'right' else x1
    else:
        return y2 if INWARD_DIR == 'down' else y1

def _axis_val_trailing(bbox) -> float:
    x1, y1, x2, y2 = bbox
    if BOUNDARY_AXIS == 'x':
        return x1 if INWARD_DIR == 'right' else x2
    else:
        return y1 if INWARD_DIR == 'down' else y2

def _centroid_axis(cx, cy) -> float:
    return cx if BOUNDARY_AXIS == 'x' else cy

def _region(val: float) -> str:
    if val < _zone_near():
        return 'near'
    elif val > _zone_far():
        return 'far'
    else:
        return 'zone'

def _inward_side() -> str:
    return 'far' if INWARD_DIR in ('right', 'down') else 'near'

def _side_to_event(destination: str) -> str:
    return 'ADD' if destination == _inward_side() else 'REMOVE'


# ── track bridge ──────────────────────────────────────────────────────────────
class TrackBridge:
    def __init__(self):
        self.lost: dict[int, dict] = {}
        self.remap: dict[int, int] = {}
        self.frame_count = 0

    def tick(self):
        self.frame_count += 1
        expired = [tid for tid, v in self.lost.items()
                   if self.frame_count - v['frame'] > RELINK_TIMEOUT]
        for tid in expired:
            self.lost.pop(tid, None)

    def mark_lost(self, tid: int, cx: float, cy: float, cls: str):
        self.lost[tid] = {'cx': cx, 'cy': cy, 'cls': cls, 'frame': self.frame_count}

    def resolve(self, tid: int, cx: float, cy: float, cls: str) -> int:
        if tid in self.remap:
            return self.remap[tid]
        best_id, best_dist = None, float('inf')
        for lost_id, v in self.lost.items():
            d = ((cx - v['cx']) ** 2 + (cy - v['cy']) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_id = d, lost_id
        if best_id is not None and best_dist < RELINK_RADIUS:
            self.remap[tid] = best_id
            self.lost.pop(best_id, None)
            print(f"[bridge] relinked #{tid} → #{best_id}  ({best_dist:.0f}px)")
            return best_id
        return tid

    def cleanup(self, active_ids: set):
        stale = [k for k, v in self.remap.items()
                 if k not in active_ids and v not in active_ids]
        for tid in stale:
            self.remap.pop(tid, None)


# ── zone state machine ────────────────────────────────────────────────────────
class ZoneTracker:
    NEAR           = 'near'
    FAR            = 'far'
    ZONE_FROM_NEAR = 'zone_from_near'
    ZONE_FROM_FAR  = 'zone_from_far'

    def __init__(self):
        self.states:        dict[int, str]   = {}
        self.classes:       dict[int, str]   = {}
        self.cooldowns:     dict[int, int]   = defaultdict(int)
        self.last_add_time: dict[str, float] = {}
        self.lost_states:   dict[int, dict]  = {}

    def _expire_lost(self):
        now = time.time()
        self.lost_states = {
            tid: v for tid, v in self.lost_states.items()
            if now - v['time'] < STATE_MEMORY_SEC
        }

    def _try_fire(self, events: list, event_type: str, cls: str, tid: int):
        if event_type == 'ADD':
            if time.time() - self.last_add_time.get(cls, 0.0) < ADD_COOLDOWN_SEC:
                return
            self.last_add_time[cls] = time.time()
        events.append((event_type, cls, tid))

    def update(self, detections: list) -> list:
        self._expire_lost()
        for tid in list(self.cooldowns):
            if self.cooldowns[tid] > 0:
                self.cooldowns[tid] -= 1

        events      = []
        current_ids = set()

        for det in detections:
            tid = det['track_id']
            if tid == -1:
                continue

            bbox  = det['bbox']
            cls   = det['class']
            cx, cy = det['centroid']
            current_ids.add(tid)
            self.classes[tid] = cls

            val        = _centroid_axis(cx, cy)
            region     = _region(val)
            prev_state = self.states.get(tid)

            if prev_state is None:
                lost = self.lost_states.pop(tid, None)
                if lost:
                    prev_state = lost['state']
                    print(f"[zone] restored state '{prev_state}' for #{tid}")

            if prev_state is None:
                if region == 'near':
                    self.states[tid] = self.NEAR
                elif region == 'far':
                    self.states[tid] = self.FAR
                else:
                    self.states[tid] = self.ZONE_FROM_NEAR
                continue

            new_state = prev_state

            if prev_state == self.NEAR:
                if region == 'zone':
                    new_state = self.ZONE_FROM_NEAR
                elif region == 'far':
                    new_state = self.FAR
                    if self.cooldowns[tid] == 0:
                        self.cooldowns[tid] = EVENT_COOLDOWN
                        self._try_fire(events, _side_to_event(self.FAR), cls, tid)

            elif prev_state == self.ZONE_FROM_NEAR:
                if region == 'near':
                    new_state = self.NEAR
                elif region == 'far':
                    new_state = self.FAR
                    if self.cooldowns[tid] == 0:
                        self.cooldowns[tid] = EVENT_COOLDOWN
                        self._try_fire(events, _side_to_event(self.FAR), cls, tid)

            elif prev_state == self.FAR:
                if region == 'zone':
                    new_state = self.ZONE_FROM_FAR
                elif region == 'near':
                    new_state = self.NEAR
                    if self.cooldowns[tid] == 0:
                        self.cooldowns[tid] = EVENT_COOLDOWN
                        self._try_fire(events, _side_to_event(self.NEAR), cls, tid)

            elif prev_state == self.ZONE_FROM_FAR:
                if region == 'far':
                    new_state = self.FAR
                elif region == 'near':
                    new_state = self.NEAR
                    if self.cooldowns[tid] == 0:
                        self.cooldowns[tid] = EVENT_COOLDOWN
                        self._try_fire(events, _side_to_event(self.NEAR), cls, tid)

            self.states[tid] = new_state

        gone = set(self.states) - current_ids
        for tid in gone:
            state = self.states.pop(tid)
            cls   = self.classes.pop(tid, 'unknown')
            self.lost_states[tid] = {'state': state, 'cls': cls, 'time': time.time()}
            self.cooldowns.pop(tid, None)

        return events

    def cleanup(self, active_ids: set):
        gone = set(self.states) - active_ids
        for tid in gone:
            self.states.pop(tid, None)
            self.classes.pop(tid, None)

    def get_state(self, tid: int) -> str:
        return self.states.get(tid, '?')


# ── preprocessing ─────────────────────────────────────────────────────────────
def resize_frame(frame):
    return cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT),
                      interpolation=cv2.INTER_LINEAR)

def apply_clahe(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def check_sharpness(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score > SHARPNESS_THRESHOLD


# ── camera motion detection ───────────────────────────────────────────────────
_prev_gray_for_motion = None

def check_camera_motion(frame) -> tuple[bool, float]:
    global _prev_gray_for_motion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _prev_gray_for_motion is None:
        _prev_gray_for_motion = gray
        return False, 0.0

    pts = cv2.goodFeaturesToTrack(_prev_gray_for_motion, maxCorners=80,
                                   qualityLevel=0.2, minDistance=20)
    if pts is None or len(pts) < 8:
        _prev_gray_for_motion = gray
        return False, 0.0

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        _prev_gray_for_motion, gray, pts, None,
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    good_old = pts[status == 1]
    good_new = new_pts[status == 1]

    if len(good_old) < 6:
        _prev_gray_for_motion = gray
        return False, 0.0

    flow = good_new - good_old
    median_dx = float(np.median(np.abs(flow[:, 0])))
    median_dy = float(np.median(np.abs(flow[:, 1])))
    magnitude = max(median_dx / INFERENCE_WIDTH, median_dy / INFERENCE_HEIGHT)

    _prev_gray_for_motion = gray
    return magnitude > CAM_MOTION_THRESHOLD, magnitude


# ── detection filter ──────────────────────────────────────────────────────────
def passes_filters(box, class_id):
    if class_id in BLACKLIST_CLASSES:
        return False, "blacklisted"
    if class_id not in ALLOWED_CLASSES:
        return False, "not allowed"
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
    w, h = x2 - x1, y2 - y1
    area = w * h
    if area < MIN_BBOX_AREA:
        return False, f"small ({area}px²)"
    if area > MAX_BBOX_AREA:
        return False, f"large ({area}px²)"
    if (h / max(w, 1)) < MIN_ASPECT_RATIO:
        return False, f"wide ({h/max(w,1):.2f})"
    return True, "ok"


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(frame):
    results    = model.track(frame, conf=CONFIDENCE_THRESHOLD,
                              persist=True, tracker=_os.path.join(_VISION_DIR, "bytetrack.yaml"),
                              verbose=False)
    detections = []
    rejected   = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id   = int(box.cls[0])
            ok, reason = passes_filters(box, class_id)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            raw_class  = model.names[class_id]
            mapped_class = CLASS_REMAP.get(raw_class, raw_class)
            if not ok:
                rejected.append({'bbox': (x1, y1, x2, y2),
                                  'reason': reason,
                                  'class': mapped_class})
                continue
            track_id   = int(box.id[0]) if box.id is not None else -1
            confidence = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                'class':      mapped_class,
                'track_id':   track_id,
                'confidence': confidence,
                'bbox':       (x1, y1, x2, y2),
                'centroid':   (cx, cy),
                'bbox_width': x2 - x1,
                'area':       (x2 - x1) * (y2 - y1),
            })
    return detections, rejected


# ── drawing ───────────────────────────────────────────────────────────────────
STATE_COLORS = {
    ZoneTracker.NEAR:           (160, 160, 160),
    ZoneTracker.FAR:            (0,   200, 100),
    ZoneTracker.ZONE_FROM_NEAR: (0,   200, 255),
    ZoneTracker.ZONE_FROM_FAR:  (0,   140, 255),
}

def draw_boundary(frame):
    line  = int(_line())
    znear = int(_zone_near())
    zfar  = int(_zone_far())

    if BOUNDARY_AXIS == 'x':
        overlay = frame.copy()
        cv2.rectangle(overlay, (znear, 0), (zfar, INFERENCE_HEIGHT), (0, 200, 255), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.line(frame, (znear, 0), (znear, INFERENCE_HEIGHT), (0, 200, 255), 1)
        cv2.line(frame, (zfar,  0), (zfar,  INFERENCE_HEIGHT), (0, 200, 255), 1)
        cv2.line(frame, (line, 0), (line, INFERENCE_HEIGHT), (0, 200, 255), 2)
        cv2.putText(frame, "zone", (line + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, znear), (INFERENCE_WIDTH, zfar), (0, 200, 255), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.line(frame, (0, znear), (INFERENCE_WIDTH, znear), (0, 200, 255), 1)
        cv2.line(frame, (0, zfar),  (INFERENCE_WIDTH, zfar),  (0, 200, 255), 1)
        cv2.line(frame, (0, line),  (INFERENCE_WIDTH, line),  (0, 200, 255), 2)
        cv2.putText(frame, "zone", (8, line - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

def draw_event_log(frame, event_log: list):
    x_start = INFERENCE_WIDTH - 230
    for i, (ev_type, ev_class, ev_tid, ev_time) in enumerate(reversed(event_log)):
        age   = time.time() - ev_time
        color = (0, 255, 120) if ev_type == 'ADD' else (0, 80, 255)
        fade  = max(0.3, 1.0 - age / 8.0)
        text  = f"{ev_type}  #{ev_tid}  {ev_class}"
        y     = 22 + i * 20
        cv2.putText(frame, text, (x_start + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)
        cv2.putText(frame, text, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    tuple(int(c * fade) for c in color), 1)

def draw_frame(frame, detections, rejected, zone_tracker,
               event_log, sharpness, fps, show_rejected):
    draw_boundary(frame)

    if show_rejected:
        for r in rejected:
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 140), 1)
            cv2.putText(frame, f"{r['class']}: {r['reason']}", (x1, y2 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 180), 1)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy          = det['centroid']
        tid             = det['track_id']
        state           = zone_tracker.get_state(tid)
        color           = STATE_COLORS.get(state, (0, 255, 0))
        label           = f"#{tid} {det['class']} {det['confidence']:.2f} [{state}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    draw_event_log(frame, event_log)

    hud = (f"{fps:.1f}fps  |  tracking: {len(detections)}  "
           f"|  events: {len(event_log)}  |  zone: ±{ZONE_WIDTH}px")
    cv2.putText(frame, hud, (10, INFERENCE_HEIGHT - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    cv2.putText(frame, "r = toggle rejected  q = quit", (10, INFERENCE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    return frame


# ── USB camera helpers ─────────────────────────────────────────────────────────
def _list_usb_cameras(max_index: int = 8):
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    if USE_USB:
        print(f"opening USB camera index {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"failed to open usb camera index {CAMERA_INDEX}")
            cap.release()
            available = _list_usb_cameras(8)
            if available:
                print("available AVFoundation camera indexes:", available)
            else:
                print("no AVFoundation cameras were detected.")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  INFERENCE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INFERENCE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        print(f"connecting to {DROIDCAM_URL}...")
        cap = cv2.VideoCapture(DROIDCAM_URL)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("could not open camera")
        return

    print(f"connected — warming up {WARMUP_SECONDS}s...")
    time.sleep(WARMUP_SECONDS)
    for _ in range(5):
        cap.read()

    zone_tracker       = ZoneTracker()
    bridge             = TrackBridge()
    event_log          = []
    inventory          = defaultdict(int)
    prev_ids           = set()
    missed             = 0
    t_prev             = time.perf_counter()
    fps                = 0.0
    show_rejected      = True
    cam_freeze_counter = 0

    print(f"zone width: ±{ZONE_WIDTH}px  |  ADD cooldown: {ADD_COOLDOWN_SEC}s")
    print("press q to quit, r to toggle rejected overlay")

    while True:
        ret, frame = cap.read()
        if not ret:
            missed += 1
            if missed >= MAX_MISSED_FRAMES:
                print("stream dropped")
                break
            time.sleep(0.05)
            continue

        missed = 0
        frame                   = resize_frame(frame)
        frame                   = apply_clahe(frame)
        sharpness, sharp_enough = check_sharpness(frame)

        if sharp_enough:
            detections, rejected = run_inference(frame)
        else:
            detections, rejected = [], []
            cv2.putText(frame, "BLURRY", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── camera motion check ───────────────────────────────────────────
        cam_moving, motion_mag = check_camera_motion(frame)
        if cam_moving:
            cam_freeze_counter = CAM_MOTION_FREEZE_FRAMES
        elif cam_freeze_counter > 0:
            cam_freeze_counter -= 1

        cam_frozen = cam_freeze_counter > 0
        if cam_frozen:
            cv2.putText(frame, "DOOR MOVING — tracking paused",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # bridge: remap re-identified tracks
        bridge.tick()
        active_ids = set()
        for det in detections:
            tid = det['track_id']
            if tid == -1:
                continue
            cx, cy          = det['centroid']
            canonical_id    = bridge.resolve(tid, cx, cy, det['class'])
            det['track_id'] = canonical_id
            active_ids.add(canonical_id)

        for gone_id in prev_ids - active_ids:
            bridge.mark_lost(gone_id, 0, 0, 'unknown')

        bridge.cleanup(active_ids)

        # zone state machine — only when camera is stable
        events = zone_tracker.update(detections) if not cam_frozen else []
        zone_tracker.cleanup(active_ids)

        for event_type, cls, tid in events:
            inventory[cls] += (1 if event_type == 'ADD' else -1)
            event_log.append((event_type, cls, tid, time.time()))
            event_log = event_log[-MAX_EVENT_LOG:]
            print(f"\n{'─' * 50}")
            print(f"  {event_type:6s}  |  class: {cls}  |  track #{tid}")
            print(f"  inventory → {dict(inventory)}")
            print(f"{'─' * 50}\n")
            try:
                requests.post(WEBAPP_URL,
                              json={"item": cls, "event": event_type},
                              timeout=1.0)
            except requests.exceptions.RequestException as e:
                print(f"[webapp] send failed: {e}")

        prev_ids = active_ids

        now    = time.perf_counter()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        frame = draw_frame(frame, detections, rejected, zone_tracker,
                           event_log, sharpness, fps, show_rejected)
        cv2.imshow('InventTrack — perception', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            show_rejected = not show_rejected

    print("\nfinal inventory:", dict(inventory))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
