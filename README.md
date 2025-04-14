# 🚗 ADAS Simulation Project – Lane & Object Detection Integration

This repository contains a series of **7 progressive coding activities** focused on building an **Advanced Driver Assistance System (ADAS)** using computer vision and GUI programming.

---

## 📌 Description

Over the course of the following six activities, we’ve incrementally developed the core ADAS functionalities:

1. 🚣️ **Lane Detection**
2. 🚧 **Traffic Sign Recognition**
3. ⚠️ **Speed Limit Recognition**
4. 💡 **Street Lamp Detection**
5. 🧠 **Recognition & Classification for Lamps**
6. 📾 **Instrument Cluster GUI for Visualization**

Now, in **Coding Activity 7**, we bring it all together into one integrated application — processing video streams in real-time and visualizing detected ADAS elements through a graphical user interface.

---

## 🧠 Coding Activity 7 – Final Integration Project

### ✅ Objective
- Integrate all previously developed ADAS modules into one unified system.
- Simultaneously detect **lanes, traffic signs, speed limits, and street lamps** from road videos.
- Visualize detected objects dynamically on a **custom GUI dashboard**, mimicking a real-world instrument cluster.

### ✅ Problem Statement
- Use **Python** (or another language) for implementation.
- Follow **Object-Oriented Programming (OOP)** principles for clean modular design.
- Process multiple video feeds to ensure robustness of the integrated system.
- **Avoid directly copying code from external sources** — build and refine based on your own previous implementations.

---

## 💪 System Architecture Overview

Each module is treated as a separate class/component:
- `LaneDetector`
- `TrafficSignRecognizer`
- `SpeedLimitRecognizer`
- `StreetLampDetector`
- `InstrumentClusterGUI`

The final application controller initializes and invokes these modules frame-by-frame to:
- Detect
- Classify
- Visualize on the GUI

---

## 🗂️ Project Structure

```
ADAS_Project/
│
├── lane_detection/
│   └── lane_detector.py
│
├── traffic_sign_recognition/
│   └── traffic_sign_recognizer.py
│
├── speed_limit_recognition/
│   └── speed_limit_detector.py
│
├── street_lamp_detection/
│   └── street_lamp_detector.py
│
├── lamp_classification/
│   └── lamp_classifier.py
│
├── gui_instrument_cluster/
│   └── cluster_gui.py
│
├── final_integration/
│   ├── main.py
│   └── video_feeds/
│       └── test_road_footage.mp4
│
└── README.md
```

---

## 🚀 How to Run Final Activity (7)

1. Clone this repository
2. Navigate to `activity_7_final_integration/`
3. Run the integration module:

```bash
python main.py
```

4. The system will:
   - Read the video feed
   - Detect lane lines, signs, lamps
   - Show them in the custom GUI interface

---

## 🧪 Testing

- Test the integration with **multiple videos**
- Ensure your classifiers and detection models work reliably
- Add logging/print statements if debugging is needed

---

## 📚 References & Credits

While this project encourages self-implementation, the following references may help for theory:

- [OpenCV Python Docs](https://docs.opencv.org/4.x/)
- [Real-time Lane Detection](https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/)
- [PapersWithCode: Lane Detection](https://paperswithcode.com/task/lane-detection/codeless)

---

## 👨‍💻 Author

Sudhir Kumar Bera – [GitHub](https://github.com/sberaconnects)

---

