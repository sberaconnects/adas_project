# 🚧 Activity 2 – Traffic Sign Recognition System

## 📌 Description

Traffic Sign Recognition is a critical feature of Advanced Driver Assistance Systems (ADAS). While driving, a human driver may unintentionally miss important road signs. This can lead to unsafe or improper driving behavior. To assist the driver, modern ADAS systems use front-facing cameras combined with AI-based algorithms to detect and recognize traffic signs in real-time. The detected signs are displayed to the driver via the vehicle's instrument cluster, enhancing safety and awareness.

---

## 🎯 Problem Statement

Your task is to develop a Traffic Sign Recognition system using computer vision and deep learning techniques.

### 🔍 Requirements
- ✅ Use **Python** (or any other programming language)
- ✅ Use **OpenCV** for image preprocessing and visualization
- ✅ Use **Machine Learning / Deep Learning** algorithms (preferably CNN)
- ✅ Download and use a **publicly available traffic sign dataset**
- ✅ Preprocess and split the data into training, validation, and test sets
- ✅ Train a CNN model to classify traffic signs with high accuracy
- ✅ Validate and test the trained model
- ✅ Save the model for later inference use

---

## 📦 Dataset Used
- [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Provided as CSV files with image paths and class labels
- Required preprocessing to group test images by label based on `Test.csv`

---

## 🧠 Deep Learning Model

Implemented a Convolutional Neural Network (CNN) with:
- Input: 32x32 RGB images
- Layers: Conv2D, MaxPooling2D, Dropout, Dense
- Output: Softmax activation for 43 traffic sign classes
- Trained with data augmentation for generalization

---

## 🗂️ Project Structure

```
activity_2_traffic_sign_recognition/
│
├── data/
│   └── GTSRB/
│       ├── Train/
│       ├── Test.csv
│       ├── Test/0/, Test/1/, ..., Test/42/
│
├── model/
│   └── traffic_sign_cnn_model.keras
│
├── traffic_sign_trainer.py
├── traffic_sign_tester.py
├── traffic_sign_predictor.py
├── prepare_test_data.py
└── training_accuracy.png
```

---

## 🛠️ How to Run

1. Organize test images using:
```bash
python prepare_test_data.py
```
2. Train the model:
```bash
python traffic_sign_trainer.py
```
3. Evaluate accuracy and confusion matrix:
```bash
python traffic_sign_tester.py
```
4. Predict from a single image:
```bash
python traffic_sign_predictor.py
```

---

## ❗ Challenges Faced

- ❌ **TensorFlow incompatibility** with Python 3.13 → switched to Python 3.10 via `venv`
- ⚠️ **Steps-per-epoch mismatch** interrupted training → fixed by removing `steps_per_epoch`
- ❌ **Misclassification due to undertraining** → resolved with full 20-epoch training and checkpointing
- 🔄 **Test data was flat** and not foldered by class → fixed using `prepare_test_data.py`
- ❗ **Low early accuracy** → improved using data augmentation and validation tuning

---

## 📚 References
- [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [Traffic Sign Recognition with CNN](https://towardsdatascience.com/traffic-sign-recognition-using-cnn-561f2ee7d685)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 👨‍💻 Author
Sudhir Kumar Bera – [GitHub](https://github.com/sberaconnects)

---
