import cv2
import time
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# ── config ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35
SHARPNESS_THRESHOLD  = 5.0
DEVICE               = 'mps'

# ── camera source ─────────────────────────────────────────────────────────────
USE_USB      = False
DROIDCAM_URL = "http://10.165.99.206:4747/video"
CAMERA_INDEX = 1

WARMUP_SECONDS    = 1.0
MAX_MISSED_FRAMES = 30
INFERENCE_WIDTH   = 640
INFERENCE_HEIGHT  = 480

# ── detection filters ─────────────────────────────────────────────────────────
ALLOWED_CLASSES   = {39}
BLACKLIST_CLASSES = {0}
MIN_BBOX_AREA     = 3000
MAX_BBOX_AREA     = 55000
MIN_ASPECT_RATIO  = 0.5

# ── boundary config ───────────────────────────────────────────────────────────
BOUNDARY_AXIS = 'x'
BOUNDARY_POS  = 0.5
INWARD_DIR    = 'right'

# ── detection mode ────────────────────────────────────────────────────────────
DETECTION_MODE = 'side_snap'   # 'side_snap' or 'velocity'

# velocity mode tuning
CROSSING_MARGIN = 40
MIN_VELOCITY    = 4
HISTORY_LEN     = 24
SMOOTH_ALPHA    = 0.4

# side_snap mode tuning
SNAP_WINDOW_SEC  = 2.5   # seconds a vanished track is held for re-linking
ADD_COOLDOWN_SEC = 3.0   # seconds before another ADD can fire for the same class
                          # REMOVE has no cooldown — you can always take something out

# shared
EVENT_COOLDOWN = 60
MAX_EVENT_LOG  = 8

# ── track bridge ──────────────────────────────────────────────────────────────
RELINK_RADIUS  = 120
RELINK_TIMEOUT = 90
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO('yolov8n.pt')
model.to(DEVICE)
print("warming up MPS...")
model(np.zeros((INFERENCE_HEIGHT, INFERENCE_WIDTH, 3), dtype=np.uint8), verbose=False)
print("MPS ready")


# ── helpers ───────────────────────────────────────────────────────────────────
def _boundary_line() -> float:
    return (BOUNDARY_POS * INFERENCE_HEIGHT if BOUNDARY_AXIS == 'y'
            else BOUNDARY_POS * INFERENCE_WIDTH)

def _which_side(cx: float, cy: float) -> str:
    val = cy if BOUNDARY_AXIS == 'y' else cx
    return 'near' if val < _boundary_line() else 'far'

def _inward_side() -> str:
    return 'far' if INWARD_DIR in ('right', 'down') else 'near'

def _side_to_event(side: str) -> str:
    return 'ADD' if side == _inward_side() else 'REMOVE'


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

    def mark_lost(self, track_id: int, cx: float, cy: float, cls: str):
        self.lost[track_id] = {'cx': cx, 'cy': cy, 'cls': cls,
                                'frame': self.frame_count}

    def resolve(self, track_id: int, cx: float, cy: float, cls: str) -> int:
        if track_id in self.remap:
            return self.remap[track_id]
        best_id, best_dist = None, float('inf')
        for lost_id, v in self.lost.items():
            dist = ((cx - v['cx']) ** 2 + (cy - v['cy']) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_id = dist, lost_id
        if best_id is not None and best_dist < RELINK_RADIUS:
            self.remap[track_id] = best_id
            self.lost.pop(best_id, None)
            print(f"[bridge] relinked #{track_id} → #{best_id}  ({best_dist:.0f}px)")
            return best_id
        return track_id

    def cleanup(self, active_ids: set):
        stale = [k for k, v in self.remap.items()
                 if k not in active_ids and v not in active_ids]
        for tid in stale:
            self.remap.pop(tid, None)


# ── approach 1 : velocity crossing ───────────────────────────────────────────
class VelocityTracker:
    def __init__(self):
        self.histories:    dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self.smoothed:     dict[int, tuple] = {}
        self.last_side:    dict[int, str]   = {}
        self.cross_origin: dict[int, float] = {}
        self.cooldowns:    dict[int, int]   = defaultdict(int)

    def update(self, track_id: int, cx: int, cy: int):
        if self.cooldowns[track_id] > 0:
            self.cooldowns[track_id] -= 1

        if track_id not in self.smoothed:
            self.smoothed[track_id] = (float(cx), float(cy))
        else:
            sx, sy = self.smoothed[track_id]
            self.smoothed[track_id] = (
                SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * sx,
                SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * sy,
            )
        sx, sy = self.smoothed[track_id]
        self.histories[track_id].append((sx, sy))

        current_side = _which_side(sx, sy)
        prev_side    = self.last_side.get(track_id)

        if prev_side is not None and prev_side != current_side:
            self.cross_origin[track_id] = _boundary_line()

        self.last_side[track_id] = current_side

        if track_id not in self.cross_origin:
            return None

        dist_past = abs(
            (sy if BOUNDARY_AXIS == 'y' else sx) - self.cross_origin[track_id]
        )
        if dist_past < CROSSING_MARGIN:
            return None

        pts = list(self.histories[track_id])
        if len(pts) < 4:
            return None
        n  = len(pts) - 1
        vx = (pts[-1][0] - pts[0][0]) / n
        vy = (pts[-1][1] - pts[0][1]) / n
        speed = abs(vy) if BOUNDARY_AXIS == 'y' else abs(vx)
        if speed < MIN_VELOCITY:
            self.cross_origin.pop(track_id, None)
            return None

        if self.cooldowns[track_id] > 0:
            self.cross_origin.pop(track_id, None)
            return None

        self.cross_origin.pop(track_id, None)
        self.cooldowns[track_id] = EVENT_COOLDOWN
        return _side_to_event(current_side)

    def cleanup(self, active_ids: set):
        gone = set(self.histories) - active_ids
        for tid in gone:
            self.histories.pop(tid, None)
            self.smoothed.pop(tid, None)
            self.last_side.pop(tid, None)
            self.cross_origin.pop(tid, None)
            self.cooldowns.pop(tid, None)


# ── approach 2 : side snapshot ────────────────────────────────────────────────
class SideSnapTracker:
    def __init__(self):
        self.sides:          dict[int, str]   = {}
        self.classes:        dict[int, str]   = {}
        self.cooldowns:      dict[int, int]   = defaultdict(int)
        self.recently_left:  dict[str, list]  = defaultdict(list)
        self.last_add_time:  dict[str, float] = {}  # class → timestamp of last ADD

    def _expire(self):
        now = time.time()
        for side in list(self.recently_left):
            self.recently_left[side] = [
                (cls, tid, t) for cls, tid, t in self.recently_left[side]
                if now - t < SNAP_WINDOW_SEC
            ]

    def _try_fire(self, events: list, event_type: str, cls: str, tid: int):
        """
        Gate for ADD events — enforces ADD_COOLDOWN_SEC between consecutive
        ADDs of the same class. REMOVE always fires immediately.
        """
        if event_type == 'ADD':
            last = self.last_add_time.get(cls, 0.0)
            if time.time() - last < ADD_COOLDOWN_SEC:
                return   # too soon, swallow the event
            self.last_add_time[cls] = time.time()
        events.append((event_type, cls, tid))

    def update(self, detections: list) -> list:
        self._expire()
        for tid in list(self.cooldowns):
            if self.cooldowns[tid] > 0:
                self.cooldowns[tid] -= 1

        events      = []
        current_ids = set()

        for det in detections:
            tid = det['track_id']
            if tid == -1:
                continue
            cx, cy        = det['centroid']
            cls           = det['class']
            current_ids.add(tid)
            self.classes[tid]  = cls
            current_side       = _which_side(cx, cy)
            prev_side          = self.sides.get(tid)
            self.sides[tid]    = current_side

            if prev_side is None:
                # new track — check if same class recently left the other side
                other_side = 'far' if current_side == 'near' else 'near'
                match_idx  = next(
                    (i for i, (c, _, _) in enumerate(self.recently_left[other_side])
                     if c == cls),
                    None
                )
                if match_idx is not None and self.cooldowns[tid] == 0:
                    self.recently_left[other_side].pop(match_idx)
                    self.cooldowns[tid] = EVENT_COOLDOWN
                    self._try_fire(events, _side_to_event(current_side), cls, tid)

            elif prev_side != current_side and self.cooldowns[tid] == 0:
                # existing track switched sides
                self.cooldowns[tid] = EVENT_COOLDOWN
                self._try_fire(events, _side_to_event(current_side), cls, tid)

        # mark disappeared tracks
        gone = set(self.sides) - current_ids
        for tid in gone:
            side = self.sides.pop(tid)
            cls  = self.classes.pop(tid, 'unknown')
            self.recently_left[side].append((cls, tid, time.time()))
            self.cooldowns.pop(tid, None)

        return events

    def cleanup(self, active_ids: set):
        gone = set(self.sides) - active_ids
        for tid in gone:
            self.sides.pop(tid, None)
            self.classes.pop(tid, None)


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
                              persist=True, tracker="bytetrack.yaml",
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
            if not ok:
                rejected.append({'bbox': (x1, y1, x2, y2),
                                  'reason': reason,
                                  'class': model.names[class_id]})
                continue
            track_id   = int(box.id[0]) if box.id is not None else -1
            confidence = float(box.conf[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                'class':      model.names[class_id],
                'track_id':   track_id,
                'confidence': confidence,
                'bbox':       (x1, y1, x2, y2),
                'centroid':   (cx, cy),
                'bbox_width': x2 - x1,
                'area':       (x2 - x1) * (y2 - y1),
            })
    return detections, rejected


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_boundary(frame):
    if BOUNDARY_AXIS == 'y':
        y = int(BOUNDARY_POS * INFERENCE_HEIGHT)
        cv2.line(frame, (0, y), (INFERENCE_WIDTH, y), (0, 200, 255), 2)
        cv2.putText(frame, f"boundary ({DETECTION_MODE})",
                    (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    else:
        x = int(BOUNDARY_POS * INFERENCE_WIDTH)
        cv2.line(frame, (x, 0), (x, INFERENCE_HEIGHT), (0, 200, 255), 2)
        cv2.putText(frame, f"boundary ({DETECTION_MODE})",
                    (x + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

def draw_trails(frame, vel_tracker: VelocityTracker, active_ids: set):
    for tid in active_ids:
        hist = vel_tracker.histories.get(tid)
        if not hist:
            continue
        pts = [(int(x), int(y)) for x, y in hist]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            cv2.line(frame, pts[i - 1], pts[i],
                     (int(255 * alpha), int(180 * alpha), 0), 1)

def draw_side_labels(frame, snap_tracker: SideSnapTracker):
    near = sum(1 for s in snap_tracker.sides.values() if s == 'near')
    far  = sum(1 for s in snap_tracker.sides.values() if s == 'far')
    if BOUNDARY_AXIS == 'x':
        lx = int(BOUNDARY_POS * INFERENCE_WIDTH)
        cv2.putText(frame, f"outside: {near}",
                    (max(4, lx - 120), INFERENCE_HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"inside: {far}",
                    (lx + 8, INFERENCE_HEIGHT - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        ly = int(BOUNDARY_POS * INFERENCE_HEIGHT)
        cv2.putText(frame, f"near: {near}", (10, max(20, ly - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"far: {far}", (10, ly + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

def draw_event_log(frame, event_log: list):
    x_start = INFERENCE_WIDTH - 230
    for i, (ev_type, ev_class, ev_tid, ev_time) in enumerate(reversed(event_log)):
        age   = time.time() - ev_time
        color = (0, 255, 120) if ev_type == 'ADD' else (0, 80, 255)
        fade  = max(0.3, 1.0 - age / 8.0)
        text  = f"{ev_type}  #{ev_tid}  {ev_class}"
        y     = 22 + i * 20
        cv2.putText(frame, text, (x_start + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)
        cv2.putText(frame, text, (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    tuple(int(c * fade) for c in color), 1)

def draw_frame(frame, detections, rejected, vel_tracker, snap_tracker,
               active_ids, event_log, sharpness, fps, show_rejected):
    draw_boundary(frame)
    if DETECTION_MODE == 'velocity':
        draw_trails(frame, vel_tracker, active_ids)
    else:
        draw_side_labels(frame, snap_tracker)

    if show_rejected:
        for r in rejected:
            x1, y1, x2, y2 = r['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 140), 1)
            cv2.putText(frame, f"{r['class']}: {r['reason']}",
                        (x1, y2 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 180), 1)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy          = det['centroid']
        label = f"#{det['track_id']} {det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (255, 0, 0), 1)
        cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (255, 0, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    draw_event_log(frame, event_log)
    hud = (f"{fps:.1f}fps  |  mode: {DETECTION_MODE}  "
           f"|  tracking: {len(detections)}  |  events: {len(event_log)}")
    cv2.putText(frame, hud, (10, INFERENCE_HEIGHT - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    cv2.putText(frame, "r = toggle rejected  q = quit",
                (10, INFERENCE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    return frame


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    if USE_USB:
        print(f"opening USB camera index {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
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

    vel_tracker   = VelocityTracker()
    snap_tracker  = SideSnapTracker()
    bridge        = TrackBridge()
    event_log     = []
    inventory     = defaultdict(int)
    prev_ids      = set()
    missed        = 0
    t_prev        = time.perf_counter()
    fps           = 0.0
    show_rejected = True

    print(f"detection mode: {DETECTION_MODE}  |  ADD cooldown: {ADD_COOLDOWN_SEC}s")
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
            cv2.putText(frame, "BLURRY", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
            hist = vel_tracker.histories.get(gone_id)
            if hist:
                lx, ly = hist[-1]
                bridge.mark_lost(gone_id, lx, ly, 'bottle')

        bridge.cleanup(active_ids)

        events = []
        if DETECTION_MODE == 'velocity':
            for det in detections:
                ev = vel_tracker.update(det['track_id'], *det['centroid'])
                if ev:
                    events.append((ev, det['class'], det['track_id']))
            vel_tracker.cleanup(active_ids)
        else:
            events = snap_tracker.update(detections)
            snap_tracker.cleanup(active_ids)

        for event_type, cls, tid in events:
            inventory[cls] += (1 if event_type == 'ADD' else -1)
            event_log.append((event_type, cls, tid, time.time()))
            event_log = event_log[-MAX_EVENT_LOG:]
            print(f"\n{'─' * 50}")
            print(f"  {event_type:6s}  |  class: {cls}  |  track #{tid}  "
                  f"|  mode: {DETECTION_MODE}")
            print(f"  inventory → {dict(inventory)}")
            print(f"{'─' * 50}\n")

        prev_ids = active_ids

        now    = time.perf_counter()
        fps    = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        frame = draw_frame(frame, detections, rejected, vel_tracker,
                           snap_tracker, active_ids, event_log,
                           sharpness, fps, show_rejected)
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