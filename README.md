# 📦 InventTrack

**Your fridge, always accounted for.**

InventTrack is a real-time fridge inventory system built for Hackathon 2026. A phone mounted inside the fridge streams video to a laptop, where a custom-trained YOLOv8 model detects items and a zone-crossing tracker decides whether each item was put **in** or taken **out**. Every event updates a live web dashboard that shows stock levels, suggests recipes, and answers Siri voice queries about what's running low.

---

## ✨ Features

- **Custom object detection.** A YOLOv8 model fine-tuned on fridge items (currently `coke_can`, `doritos`, and `lemon`).
- **Persistent multi-object tracking.** ByteTrack with a custom config, plus a `TrackBridge` layer that re-links tracks that briefly drop out so one item doesn't get counted twice.
- **ADD / REMOVE detection.** A virtual boundary with a buffer zone splits the frame; a per-track state machine fires an event only when an item fully crosses from one side to the other.
- **Robust to real-world conditions.** CLAHE contrast enhancement for dim fridge lighting, a Laplacian sharpness check that skips blurry frames, and optical-flow camera-motion detection that pauses counting while the door is swinging.
- **False-positive filtering.** Bounding-box area and aspect-ratio filters, per-class ADD cooldowns, and per-track event cooldowns.
- **Live dashboard.** A FastAPI backend pushes inventory updates to the browser over WebSockets, with stock status (Good / Medium / Low / Empty) and an activity feed.
- **AI recipe suggestions.** GPT-4o-mini proposes recipes based on what's currently in the fridge.
- **Siri integration.** Endpoints built for Siri Shortcuts: "What am I low on?" and "Can I make *X*?"
- **Restock helper.** The dashboard builds a low-stock cart and generates a ready-to-paste prompt for Amazon's Rufus assistant.

---

## 🧠 How It Works

```
┌──────────────┐  video   ┌──────────────────────────────┐  POST /update-vision  ┌─────────────────┐  WebSocket  ┌───────────┐
│ Phone camera │ ───────▶ │ perception.py                │ ────────────────────▶ │ FastAPI backend │ ──────────▶ │ Dashboard │
│ (USB or      │          │ CLAHE → sharpness → YOLOv8   │  {item, event}        │ (main.py)       │             │ (browser) │
│  DroidCam)   │          │ → ByteTrack → zone tracker   │                       │ inventory state │             └───────────┘
└──────────────┘          └──────────────────────────────┘                       │ OpenAI · Siri   │
                                                                                 └─────────────────┘
```

1. **Capture.** Frames come from an iPhone over USB (AVFoundation) or over Wi-Fi through DroidCam, and are resized to 640×480.
2. **Preprocess.** CLAHE boosts contrast, and frames below the sharpness threshold are skipped.
3. **Detect and track.** YOLOv8 runs on Apple Silicon (`mps`) with ByteTrack assigning persistent IDs. `TrackBridge` reconnects IDs that were lost within a set radius and time window.
4. **Classify the event.** The frame is split by a vertical boundary with a ±120 px buffer zone. Each track moves through `near → zone → far` states; a full crossing toward the inward side is an **ADD** and the reverse is a **REMOVE**. Counting is frozen whenever optical flow shows the camera itself is moving.
5. **Publish.** Events are POSTed to the backend, which updates counts, recomputes stock status, and broadcasts the new state to every connected dashboard.

---

## 🗂️ Project Structure

```
InventTrack/
├── requirements.txt
└── InventTrack/
    ├── Vision/
    │   ├── perception.py          # Live detection, tracking, and ADD/REMOVE events
    │   ├── preception_static.py   # Run detection on a single image (debugging)
    │   ├── Bytetrack.yaml         # Custom ByteTrack settings
    │   ├── trained_model.pt       # Fine-tuned YOLOv8 weights
    │   └── yolov8n.pt / yolov8s.pt  # Base YOLOv8 weights
    └── WebApp/
        ├── main.py                # FastAPI server: REST, WebSocket, OpenAI, Siri
        └── webpage.html           # Dashboard UI
```

---

## 🚀 Getting Started

### Requirements

- macOS on Apple Silicon (the code uses the `mps` device and AVFoundation camera backend)
- Python 3.10+
- An iPhone connected over USB, or any phone running [DroidCam](https://www.dev47apps.com/)
- An OpenAI API key (for recipes and the Siri recipe check)

### Install

```bash
git clone https://github.com/shawnwakeman/InventTrack.git
cd InventTrack
python -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python fastapi uvicorn openai requests numpy
```

### Configure

Set your OpenAI key as an environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
```

Camera and detection settings live at the top of `Vision/perception.py`:

| Setting | Default | Purpose |
|---|---|---|
| `USE_USB` | `True` | `True` for a USB camera, `False` for DroidCam |
| `CAMERA_INDEX` | `1` | AVFoundation index of the USB camera |
| `DROIDCAM_URL` | `http://<phone-ip>:4747/video` | DroidCam stream URL |
| `DEVICE` | `mps` | Use `cuda` or `cpu` on other hardware |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum detection confidence |
| `BOUNDARY_AXIS` / `BOUNDARY_POS` | `x` / `0.5` | Where the in/out line sits |
| `INWARD_DIR` | `right` | Which direction counts as "into the fridge" |
| `ZONE_WIDTH` | `120` | Buffer zone half-width in pixels |

### Run

Start the backend first:

```bash
cd InventTrack/WebApp
python main.py
```

Open **http://localhost:8001** for the dashboard, then start the vision pipeline in a second terminal:

```bash
cd InventTrack/Vision
python perception.py
```

In the preview window, press `r` to toggle the rejected-detection overlay and `q` to quit.

To test detection on a single image:

```bash
python preception_static.py path/to/image.jpg
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/inventory` | Current inventory and the last 20 events |
| `POST` | `/update-vision` | Report an event: `{"item": "lemon", "event": "ADD"}` |
| `POST` | `/reset` | Clear inventory and event log |
| `WS` | `/ws` | Live inventory updates |
| `GET` | `/ai/recipes` | Three recipe ideas based on current inventory |
| `GET` | `/siri/low-stock` | Spoken summary of low or empty items |
| `GET` | `/siri/recipe-check?dish=...` | Whether you have the ingredients for a dish |

Stock status is based on count: `0` is Empty, `1` is Low, `2` is Medium, and `3+` is Good.

---

## 🛠️ Tech Stack

**Vision:** Ultralytics YOLOv8, ByteTrack, OpenCV, NumPy, PyTorch (MPS)
**Backend:** FastAPI, Uvicorn, WebSockets, OpenAI API (GPT-4o-mini)
**Frontend:** HTML, CSS, vanilla JavaScript
**Integrations:** Siri Shortcuts, Amazon Rufus

---

## ⚠️ Known Limitations

- The model currently recognizes three item classes. Adding more means retraining `trained_model.pt`.
- The trained model's class labels were shuffled during training, so `perception.py` corrects them with a `CLASS_REMAP` table. Retraining with fixed labels would remove the need for it.
- Inventory is held in memory and resets when the server restarts.
- The pipeline targets macOS; other platforms need `DEVICE` and the camera backend changed.

## 🔭 Future Work

- Expand the dataset to cover common groceries
- Persist inventory in a database
- Expiration-date tracking and restock notifications
- Run inference on-device on the phone

---

Built by [Shawn Wakeman](https://github.com/shawnwakeman) for Hackathon 2026.
