# 🎯 Object Detection with YOLOv3

In this project, we utilize the YOLOv3 object detection algorithm to detect important road-related objects such as cars, buses, pedestrians, bicycles, street lamps, and traffic signs using camera images. This forms part of an Advanced Driver-Assistance Systems (ADAS) framework.

---

## 📌 Description

Reliable object detection is crucial for automated vehicle systems to identify potential hazards on the road accurately. YOLOv3 ("You Only Look Once" v3) provides an efficient real-time object detection solution suitable for ADAS applications.

**This project will:**
1. Download and configure the YOLOv3 model.
2. Apply the model to detect important road-related objects.
3. Visualize detections with bounding boxes around detected objects.

---

## 🗂️ Project Structure

```
object_detection_yolov3/
├── scripts/
│   ├── download_yolo_weights.py   # Script to fetch YOLO model files
│   └── object_detection.py        # Main script for detection
│
├── README.md                      # This documentation

Centralized directories:
data/
└── images/                        # Sample images for object detection
    ├── road1.jpg
    ├── road2.jpg
    └── ...

model/
└── yolov3/                        # YOLO model files
    ├── yolov3.cfg
    ├── yolov3.weights
    └── coco.names
```

---

## 🚀 Getting Started

### 1. Download YOLOv3 Model
```bash
python scripts/download_yolo_weights.py
```

### 2. Run Object Detection
```bash
python scripts/object_detection.py \
  --images ../../data/images/road1.jpg ../../data/images/road2.jpg \
  --conf-thres 0.5 --nms-thres 0.4
```

---

## 📦 Dependencies

Install dependencies:
```bash
pip install opencv-python numpy argparse
```

---

## 🔄 Filtering

In `object_detection.py`, specify classes relevant for road object detection:
```python
filter_classes = ['car', 'bus', 'person', 'bicycle', 'traffic light', 'stop sign', 'street lamp']
```

---

## 📚 References

- YOLOv3 paper: https://arxiv.org/abs/1506.02640
- YOLO Official Site: https://pjreddie.com/darknet/yolo/
- OpenCV DNN Module: https://docs.opencv.org/master/d6/d0f/group__dnn.html

---

## 👨‍💻 Author
Sudhir Kumar Bera – [GitHub](https://github.com/sberaconnects)

