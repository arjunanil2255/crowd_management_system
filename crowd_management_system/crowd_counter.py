import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ✅ Completely hide GPU from PyTorch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
from datetime import datetime
import time

print("=" * 60)
print("🔥 DEVICE CHECK")
print("Running on CPU (stable mode)")
print("=" * 60)

# Load YOLOv8 model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
model.to("cpu")
print("✅ Model running on: CPU")

# ✅ DeepSORT with mobilenet on CPU
tracker = DeepSort(
    max_age=30,
    n_init=5,
    max_iou_distance=0.7,
    embedder="mobilenet",
    embedder_gpu=False,   # ✅ Force CPU for embedder
    half=False
)
print("✅ DeepSORT tracker initialized!")

# ── Config ───────────────────────────────────────────────────────
CONFIG_FILE = 'config.json'
SAVE_INTERVAL = 5
last_save_time = time.time()
last_config_check = time.time()


def get_camera_source():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            source = config.get('camera_source', '0')
            if str(source).isdigit():
                return int(source)
            return source
    except:
        return 0


def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "location_name": "Canteen",
                "max_capacity": 20,
                "camera_source": "0"
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"location_name": "Canteen", "max_capacity": 20, "camera_source": "0"}


# ── Connect Camera ───────────────────────────────────────────────
camera_source = get_camera_source()
cap = cv2.VideoCapture(camera_source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 120)

# ── Load Config ──────────────────────────────────────────────────
config = load_config()
LOCATION_NAME = config.get("location_name", "Canteen")
MAX_CAPACITY = config.get("max_capacity", 20)

# ── Tracking Variables ───────────────────────────────────────────
tracked_ids = set()
total_entered = 0
frame_count = 0
tracks = []
person_count = 0

print("=" * 60)
print(f"Starting crowd monitoring for: {LOCATION_NAME}")
print(f"Maximum Capacity: {MAX_CAPACITY} people")
print(f"Camera Source: {camera_source}")
print("DeepSORT tracking: ENABLED")
print("Press 'q' to quit")
print("=" * 60)

while True:

    # ── Auto-reconnect if camera changed ────────────────────────
    new_source = get_camera_source()
    if new_source != camera_source:
        print(f"📷 Camera source changed to: {new_source}")
        cap.release()
        camera_source = new_source
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(3, 640)
        cap.set(4, 480)
        cap.set(10, 120)
        print(f"✅ Switched to new camera!")

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera, retrying...")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    frame_count += 1
    current_time = time.time()

    # ── Reload Config every 10 seconds ──────────────────────────
    if current_time - last_config_check >= 10:
        config = load_config()
        LOCATION_NAME = config.get("location_name", LOCATION_NAME)
        MAX_CAPACITY = config.get("max_capacity", MAX_CAPACITY)
        last_config_check = current_time
        print(f"⚙️  Config reloaded: {LOCATION_NAME}, Capacity: {MAX_CAPACITY}")

    # ── YOLO Detection every 2nd frame ──────────────────────────
    if frame_count % 2 == 1:
        results = model(
            frame,
            verbose=False,
            classes=[0],
            conf=0.35,
            imgsz=640,
            device="cpu"
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])

                if h > 20 and w > 15 and conf > 0.35:
                    detections.append(([x1, y1, w, h], conf, 0))

        # ── DeepSORT Tracking ────────────────────────────────────
        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"Tracker error: {e}")
            tracks = []

        # Count confirmed tracks
        person_count = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            person_count += 1
            if track.track_id not in tracked_ids:
                tracked_ids.add(track.track_id)
                total_entered += 1

    # ── Draw Boxes on Every Frame ────────────────────────────────
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        color_index = track_id % 5
        colors = [
            (0, 255, 0),
            (255, 165, 0),
            (0, 200, 255),
            (255, 0, 255),
            (0, 255, 128),
        ]
        color = colors[color_index]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame,
                      (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 5, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # ── Occupancy ────────────────────────────────────────────────
    occupancy_percent = (person_count / MAX_CAPACITY) * 100 if MAX_CAPACITY > 0 else 0

    if occupancy_percent >= 100:
        status = "Full"
        color = (0, 0, 255)
    elif occupancy_percent >= 75:
        status = "Almost Full"
        color = (0, 140, 255)
    elif occupancy_percent >= 50:
        status = "Half Full"
        color = (0, 255, 255)
    elif occupancy_percent >= 25:
        status = "Quarter Full"
        color = (0, 255, 0)
    else:
        status = "Empty"
        color = (0, 255, 0)

    # ── Display Info on Frame ────────────────────────────────────
    cv2.putText(frame, f"People: {person_count}/{MAX_CAPACITY} - {status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Occupancy: {occupancy_percent:.1f}%",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Location: {LOCATION_NAME}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Entered: {total_entered}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    # ── Capacity Bar ─────────────────────────────────────────────
    bar_width = 600
    bar_height = 30
    bar_x = 20
    bar_y = frame.shape[0] - 50

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)

    filled_width = min(int((bar_width * occupancy_percent) / 100), bar_width)

    if occupancy_percent >= 100:
        bar_color = (0, 0, 255)
    elif occupancy_percent >= 75:
        bar_color = (0, 140, 255)
    elif occupancy_percent >= 50:
        bar_color = (0, 255, 255)
    else:
        bar_color = (0, 255, 0)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)

    # ── Save Data ────────────────────────────────────────────────
    if current_time - last_save_time >= SAVE_INTERVAL:
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": LOCATION_NAME,
            "count": person_count,
            "max_capacity": MAX_CAPACITY,
            "occupancy_percent": round(occupancy_percent, 1),
            "status": status,
            "total_entered": total_entered
        }
        try:
            with open('crowd_data.json', 'a') as f:
                f.write(json.dumps(data) + '\n')
            print(f"✓ Saved: {person_count}/{MAX_CAPACITY} people "
                  f"({occupancy_percent:.1f}%) - {status} | "
                  f"Total entered: {total_entered}")
        except Exception as e:
            print(f"✗ Error saving: {e}")

        last_save_time = current_time

    cv2.imshow('Crowd Counter - DeepSORT Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print("Monitoring stopped.")
print(f"Total unique people tracked: {total_entered}")
print("=" * 60)