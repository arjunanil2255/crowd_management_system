import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import time
import os
import threading
import numpy as np
from datetime import datetime

print("Loading YOLO model...")
model = YOLO('yolov8l.pt')
print("✅ YOLO loaded!")

CONFIG_FILE = 'config.json'
SAVE_INTERVAL = 5

# ── Shared Data Between Cameras ─────────────────────────────────
# Populated dynamically based on config
camera_data = {}
data_lock = threading.Lock()


def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {
            "location_name": "Canteen",
            "max_capacity": 20,
            "cameras": [
                {"id": 1, "name": "Camera 1", "url": "0", "enabled": True},
                {"id": 2, "name": "Camera 2", "url": "1", "enabled": True}
            ]
        }


def get_color_histogram(frame, bbox):
    """Extract color histogram from person's bounding box - used for Re-ID"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except:
        return None


def compare_histograms(hist1, hist2):
    """Compare two histograms - returns similarity score 0 to 1"""
    if hist1 is None or hist2 is None:
        return 0
    try:
        score = cv2.compareHist(
            hist1.reshape(-1, 1).astype(np.float32),
            hist2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        return max(0, score)
    except:
        return 0


def is_duplicate(hist, current_camera_id, threshold=0.85):
    """Check if person already counted in another camera"""
    with data_lock:
        for cam_id, cam_info in camera_data.items():
            if cam_id == current_camera_id:
                continue
            for track_id, other_hist in cam_info['tracks'].items():
                similarity = compare_histograms(hist, other_hist)
                if similarity > threshold:
                    return True
    return False


def camera_thread(camera_id, camera_url, camera_name):
    """Each camera runs in its own thread"""

    print(f"📷 Starting {camera_name}...")

    tracker = DeepSort(
        max_age=30,
        n_init=5,
        max_iou_distance=0.7
    )

    source = int(camera_url) if str(camera_url).isdigit() else camera_url
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Cannot open {camera_name}")
        return

    print(f"✅ {camera_name} connected!")

    frame_count = 0
    tracks = []
    local_count = 0
    current_tracks = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ {camera_name} lost connection, retrying...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_count += 1

        if frame_count % 2 == 1:
            # YOLO Detection
            results = model(frame, verbose=False, classes=[0], conf=0.5)

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])

                    if h > 50 and w > 30 and conf > 0.5:
                        detections.append(([x1, y1, w, h], conf, 0))

            # DeepSORT Tracking
            tracks = tracker.update_tracks(detections, frame=frame)

            # Recalculate local_count and current_tracks
            local_count = 0
            current_tracks_temp = {}
            for track in tracks:
                if track.is_confirmed():
                    local_count += 1
                    ltrb = track.to_ltrb()
                    hist = get_color_histogram(frame, ltrb)
                    current_tracks_temp[int(track.track_id)] = hist

            current_tracks = current_tracks_temp

        # Render tracks on frame
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            hist = current_tracks.get(track_id)
            duplicate = is_duplicate(hist, camera_id)

            if duplicate:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID:{track_id} [DUP]",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                color_index = track_id % 5
                colors = [(0, 255, 0), (255, 165, 0), (0, 200, 255),
                          (255, 0, 255), (0, 255, 128)]
                color = colors[color_index]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update shared camera data
        with data_lock:
            camera_data[camera_id]['tracks'] = current_tracks
            camera_data[camera_id]['count'] = local_count

        # Show camera window
        cv2.putText(frame, f"{camera_name}: {local_count} people",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "GREEN=Counted  RED=Duplicate",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(f'{camera_name}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


def save_combined_data(location_name, max_capacity, last_save_time):
    """Save combined count from all cameras"""
    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL:
        with data_lock:
            total_count = sum(camera_data[cam]['count'] for cam in camera_data)
            # Build camera counts dynamically
            cam_counts = {
                f"camera_{cam}_count": camera_data[cam]['count']
                for cam in camera_data
            }

        occupancy_percent = (total_count / max_capacity) * 100 if max_capacity > 0 else 0

        if occupancy_percent >= 100:
            status = "Full"
        elif occupancy_percent >= 75:
            status = "Almost Full"
        elif occupancy_percent >= 50:
            status = "Half Full"
        elif occupancy_percent >= 25:
            status = "Quarter Full"
        else:
            status = "Empty"

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location_name,
            "count": total_count,
            "max_capacity": max_capacity,
            "occupancy_percent": round(occupancy_percent, 1),
            "status": status,
            **cam_counts  # adds camera_1_count, camera_2_count etc dynamically
        }

        try:
            with open('crowd_data.json', 'a') as f:
                f.write(json.dumps(data) + '\n')
            # Dynamic print for any number of cameras
            cam_info = " | ".join([
                f"Cam{cam}: {camera_data[cam]['count']}"
                for cam in camera_data
            ])
            print(f"✓ Total: {total_count} people ({cam_info}) - {status}")
        except Exception as e:
            print(f"✗ Error saving: {e}")

        return current_time
    return last_save_time


# ── Main ─────────────────────────────────────────────────────────
config = load_config()
location_name = config.get("location_name", "Canteen")
max_capacity = config.get("max_capacity", 20)
cameras = config.get("cameras", [])

# ✅ Initialize camera_data dynamically based on enabled cameras
for cam in cameras:
    if cam.get("enabled", True):
        camera_data[cam['id']] = {'tracks': {}, 'count': 0, 'frame': None}

print("=" * 60)
print(f"Location: {location_name}")
print(f"Max Capacity: {max_capacity}")
print(f"Cameras: {len(cameras)}")
print("GREEN box = Counted | RED box = Duplicate (not counted)")
print("=" * 60)

if len(cameras) == 0:
    print("❌ No cameras found in config.json!")
    print("👉 Go to settings page and add cameras first.")
    exit()

# Start each camera in its own thread
threads = []
for cam in cameras:
    if cam.get("enabled", True):
        t = threading.Thread(
            target=camera_thread,
            args=(cam['id'], cam['url'], cam['name']),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(1)

last_save_time = time.time()
print("\n✅ All cameras running! Press Q in any camera window to quit.\n")

try:
    while True:
        last_save_time = save_combined_data(
            location_name, max_capacity, last_save_time
        )
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")

cv2.destroyAllWindows()
print("=" * 60)
print("Multi-camera monitoring stopped.")
print("=" * 60)