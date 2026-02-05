import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import time
import os

# Load YOLOv8 model (downloads automatically on first run)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # 'n' = nano (fastest, smallest)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(10, 120)  # Brightness

# Configuration file
CONFIG_FILE = 'config.json'
SAVE_INTERVAL = 5  # Save data every 5 seconds
last_save_time = time.time()
last_config_check = time.time()


def load_config():
    """Load configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "location_name": "Canteen",
                "max_capacity": 20
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"location_name": "Canteen", "max_capacity": 20}


# Load initial configuration
config = load_config()
LOCATION_NAME = config.get("location_name", "Canteen")
MAX_CAPACITY = config.get("max_capacity", 20)

print("=" * 60)
print(f"Starting crowd monitoring for: {LOCATION_NAME}")
print(f"Maximum Capacity: {MAX_CAPACITY} people")
print("Configuration updates automatically every 10 seconds")
print("Press 'q' to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read from webcam")
        break

    # Check for configuration updates every 10 seconds
    current_time = time.time()
    if current_time - last_config_check >= 10:
        config = load_config()
        LOCATION_NAME = config.get("location_name", LOCATION_NAME)
        MAX_CAPACITY = config.get("max_capacity", MAX_CAPACITY)
        last_config_check = current_time
        print(f"⚙️  Config reloaded: {LOCATION_NAME}, Capacity: {MAX_CAPACITY}")

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Count people (class 0 = person in COCO dataset)
    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Person class
                person_count += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add confidence score
                confidence = float(box.conf[0])
                label = f"Person {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate occupancy percentage
    occupancy_percent = (person_count / MAX_CAPACITY) * 100 if MAX_CAPACITY > 0 else 0

    # Determine status based on capacity percentage
    if occupancy_percent >= 100:
        status = "Full"
        color = (0, 0, 255)  # Red
    elif occupancy_percent >= 75:
        status = "Almost Full"
        color = (0, 140, 255)  # Orange
    elif occupancy_percent >= 50:
        status = "Half Full"
        color = (0, 255, 255)  # Yellow
    elif occupancy_percent >= 25:
        status = "Quarter Full"
        color = (0, 255, 0)  # Green
    else:
        status = "Empty"
        color = (0, 255, 0)  # Green

    # Display count with status
    cv2.putText(frame, f"People: {person_count}/{MAX_CAPACITY} - {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Occupancy: {occupancy_percent:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame, f"Location: {LOCATION_NAME}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw capacity bar at bottom
    bar_width = 600
    bar_height = 30
    bar_x = 20
    bar_y = frame.shape[0] - 50

    # Background bar (gray)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (200, 200, 200), -1)

    # Filled bar based on occupancy
    filled_width = int((bar_width * occupancy_percent) / 100)
    if filled_width > bar_width:
        filled_width = bar_width

    # Choose bar color based on percentage
    if occupancy_percent >= 100:
        bar_color = (0, 0, 255)  # Red
    elif occupancy_percent >= 75:
        bar_color = (0, 140, 255)  # Orange
    elif occupancy_percent >= 50:
        bar_color = (0, 255, 255)  # Yellow
    else:
        bar_color = (0, 255, 0)  # Green

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                  bar_color, -1)

    # Border around bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (0, 0, 0), 2)

    # Save data every SAVE_INTERVAL seconds
    if current_time - last_save_time >= SAVE_INTERVAL:
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": LOCATION_NAME,
            "count": person_count,
            "max_capacity": MAX_CAPACITY,
            "occupancy_percent": round(occupancy_percent, 1),
            "status": status
        }

        # Append to JSON file
        try:
            with open('crowd_data.json', 'a') as f:
                f.write(json.dumps(data) + '\n')
            print(f"✓ Saved: {person_count}/{MAX_CAPACITY} people ({occupancy_percent:.1f}%) - {status}")
        except Exception as e:
            print(f"✗ Error saving data: {e}")

        last_save_time = current_time

    # Display the frame
    cv2.imshow('Crowd Counter - Press Q to Quit', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print("Monitoring stopped.")
print("Data saved to: crowd_data.json")
print("You can now run app.py to view the dashboard!")
print("=" * 60)