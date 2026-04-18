import cv2
import numpy as np
from ultralytics import YOLO

# ── config ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
SHARPNESS_THRESHOLD  = 10.0
DEVICE               = 'mps'  # Apple Silicon GPU
# ────────────────────────────────────────────────────

model = YOLO('yolov8n.pt')  # downloads COCO weights automatically first run
model.to(DEVICE)

def apply_clahe(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l       = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def check_sharpness(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score > SHARPNESS_THRESHOLD

def run_inference(frame):
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append({
                'class':      class_name,
                'confidence': confidence,
                'bbox':       (x1, y1, x2, y2),
                'centroid':   (cx, cy),
                'bbox_width': x2 - x1,
            })

    return detections

def draw_detections(frame, detections, sharpness_score):
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy          = det['centroid']
        label           = f"{det['class']} {det['confidence']:.2f}"

        # bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # centroid dot
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

        # crosshair — shows exactly where servo loop would target
        cv2.line(frame, (cx-15,cy), (cx+15,cy), (255,0,0), 1)
        cv2.line(frame, (cx,cy-15), (cx,cy+15), (255,0,0), 1)

        # label
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # sharpness score in corner — so you can tune the threshold
    cv2.putText(frame, f"sharpness: {sharpness_score:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("could not open webcam")
        return

    print("running — press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocessing
        frame     = apply_clahe(frame)
        sharpness, sharp_enough = check_sharpness(frame)

        if sharp_enough:
            detections = run_inference(frame)
        else:
            # frame too blurry — skip inference, hold last result
            detections = []
            cv2.putText(frame, "BLURRY — skipping", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        frame = draw_detections(frame, detections, sharpness)
        cv2.imshow('perception pipeline', frame)

        # print detections to terminal so you can see raw output
        for det in detections:
            print(f"{det['class']:15s} conf={det['confidence']:.2f}  "
                  f"centroid={det['centroid']}  bbox_width={det['bbox_width']}px")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()