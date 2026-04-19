import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ── config ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.1
SHARPNESS_THRESHOLD  = 100.0
DEVICE               = 'mps'
# ────────────────────────────────────────────────────

model = YOLO('yolov8n.pt')
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
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
        cv2.line(frame, (cx-15,cy), (cx+15,cy), (255,0,0), 1)
        cv2.line(frame, (cx,cy-15), (cx,cy+15), (255,0,0), 1)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # detection summary in top left
    y_offset = 30
    cv2.putText(frame, f"sharpness: {sharpness_score:.1f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    y_offset += 25
    cv2.putText(frame, f"detections: {len(detections)}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
    for det in detections:
        y_offset += 25
        summary = f"  {det['class']} {det['confidence']:.2f}  centroid={det['centroid']}"
        cv2.putText(frame, summary, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    return frame

def main():
    # get image path from command line or prompt
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("image path: ").strip()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"could not load image: {image_path}")
        return

    frame      = apply_clahe(frame)
    sharpness, sharp_enough = check_sharpness(frame)

    if sharp_enough:
        detections = run_inference(frame)
    else:
        print(f"warning: low sharpness score {sharpness:.1f} — detections may be unreliable")
        detections = run_inference(frame)  # run anyway on static image

    frame = draw_detections(frame, detections, sharpness)

    # print to terminal
    print(f"\nsharpness: {sharpness:.1f}")
    print(f"detections: {len(detections)}")
    for det in detections:
        print(f"  {det['class']:15s} conf={det['confidence']:.2f}  "
              f"centroid={det['centroid']}  bbox_width={det['bbox_width']}px")

    # save output image alongside the input
    output_path = image_path.rsplit('.', 1)[0] + '_annotated.jpg'
    cv2.imwrite(output_path, frame)
    print(f"\nsaved to {output_path}")

    # show window — press any key to close
    cv2.imshow('perception — static', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()