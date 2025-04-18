import os
import urllib.request

# Centralized path to save YOLO files
MODEL_DIR = os.path.join("..", "model", "yolov3")
os.makedirs(MODEL_DIR, exist_ok=True)

# URLs to YOLOv3 files
YOLO_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Target file paths
cfg_path = os.path.join(MODEL_DIR, "yolov3.cfg")
weights_path = os.path.join(MODEL_DIR, "yolov3.weights")
names_path = os.path.join(MODEL_DIR, "coco.names")

# Download helper


def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(destination)}...")
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Saved to {destination}")
    else:
        print(f"✔️ {os.path.basename(destination)} already exists.")


# Run downloads
download_file(YOLO_CFG_URL, cfg_path)
download_file(YOLO_WEIGHTS_URL, weights_path)
download_file(COCO_NAMES_URL, names_path)

print("\n✅ All YOLOv3 files are ready in:", MODEL_DIR)
