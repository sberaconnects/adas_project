# 🛣️ Lane Detection System for ADAS Application

**Coding Activity 1 – Development of Lane Detection System for ADAS Application**

---

## 📌 Description

This project implements a basic **Lane Detection System** using computer vision techniques. The system uses video footage from the front camera of an ego vehicle to detect lane lines and determine the lane the vehicle is driving in. Lane detection is a foundational feature in Advanced Driver Assistance Systems (ADAS), enabling functionalities such as:

- 🚗 Lane Departure Warning (LDW)  
- 🚧 Lane Change Assist (LCA)  
- 🔄 Lane Keep Assist (LKA)

---

## 🚨 Problem Statement

Develop a working lane detection system that can continuously identify road lanes in video frames using the following constraints:

- ✅ Use **Python** (or any other programming language)
- ✅ Utilize **OpenCV** for image processing and lane line detection
- ✅ Apply **Machine Learning / Deep Learning** as needed for advanced improvements
- ✅ Download and use videos from public datasets such as **KITTI** or **nuScenes**
- ✅ Validate the system by running it continuously on the selected road video

---

## 🎯 Goals

- Detect visible road lane markings from video
- Highlight detected lanes in the video frames
- Optionally determine the lane in which the ego vehicle is moving

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python     | Development language |
| OpenCV     | Computer vision & image processing |
| NumPy      | Numerical operations |
| FFmpeg     | Video stream handling (for merging streams if needed) |

---

## 📥 Dataset & Video Source

- Download a KITTI dataset sequence: [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- OR use a publicly available video from YouTube simulating front camera driving

---

## 🔗 References & Learning Resources

- [Real-time Lane Detection using OpenCV – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/)
- [Simple Lane Detection with OpenCV – Medium](https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0)
- [PapersWithCode - Lane Detection](https://paperswithcode.com/task/lane-detection/codeless)
- [Ultra Fast Structure-aware Deep Lane Detection (arXiv)](https://arxiv.org/pdf/1903.02193.pdf)

---

## 📸 Sample Output

*To be updated with screenshots or output video frames.*

---

## 📦 How to Run

1. Clone this repo or copy the script
2. Place your road video (`kitti_road_video.mp4`) in the same directory
3. Run the Python script:

```bash
python lane_detector.py

