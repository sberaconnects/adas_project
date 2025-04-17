# 🚦 Activity 3 – Speed Limit Recognition System

This activity focuses on detecting **speed limit signs** (20 km/h through 120 km/h) using a Convolutional Neural Network on the **TrafficSign speed-limit subset**.

---

## 📌 Description

ADAS systems can warn drivers when they exceed legal speed limits. In this activity, we reuse the **speed-limit** classes (IDs 0–7) from the full TrafficSign dataset to train a model that recognizes only speed‑limit signs.

### Centralized Data & Models
- **Data source:** `data/TrafficSign/` (full dataset) and `data/SpeedLimit/` (subset for speed-limit)  
- **Models:** all trained models stored under `models/speed_limit/`

---

## 🎯 Problem Statement

- Use **Python** and **OpenCV** for image processing.  
- Implement a **CNN** (or reuse the architecture from Activity 2) for speed‑limit classification.  
- Prepare the subset by extracting classes 0–7 from TrafficSign:  
  ```bash
  python prepare_speed_limit_data.py
  ```
- Train the model, validate on a held‑out set, and save the best checkpoint.  
- Evaluate on the test set and save performance metrics.  
- Build a predictor for single‑image inference.

---

## 🗂️ Project Structure

```
activity_3_speed_limit_recognition/
├── prepare_speed_limit_data.py       # create data/SpeedLimit from data/TrafficSign
├── speed_limit_trainer.py            # trains model; checkpoints to models/speed_limit/
├── speed_limit_tester.py             # evaluates test accuracy; confusion matrix
├── speed_limit_predictor.py          # predicts a single image; displays result
└── README.md                         # this file
```

---

## 📦 Prerequisites

```bash
# from project root
pip install -r requirements.txt
```

- Python 3.10 (via `venv`)  
- TensorFlow 2.10–2.13  
- OpenCV, NumPy, Matplotlib, Scikit-learn, Pandas  

---

## 🚀 Step-by-Step

1. **Prepare data**:
   ```bash
   python prepare_speed_limit_data.py
   ```
   - Reads from `data/TrafficSign/Train/0–7` and `data/TrafficSign/Test/0–7`  
   - Writes into `data/SpeedLimit/Train/0–7` and `data/SpeedLimit/Test/0–7`

2. **Train model**:
   ```bash
   python speed_limit_trainer.py
   ```
   - Uses `data/SpeedLimit/Train` & `data/SpeedLimit/Test`  
   - Saves best model: `models/speed_limit/best_speed_limit_model.keras`  
   - Final model: `models/speed_limit/speed_limit_model.keras`  
   - Training plot: `training_accuracy.png`

3. **Test model**:
   ```bash
   python speed_limit_tester.py
   ```
   - Loads `models/speed_limit/best_speed_limit_model.keras`  
   - Outputs classification report and saves `confusion_matrix.png` in current folder

4. **Predict on image**:
   ```bash
   python speed_limit_predictor.py
   ```
   - Prompts for image path (e.g., `data/SpeedLimit/Test/3/00003.png`)  
   - Displays predicted speed limit class and confidence

---

## ❗ Challenges & Notes

- **Centralized paths** simplify maintenance but require consistent file references.  
- **Underfitting risk**: if accuracy is low, increase epochs or augment more aggressively.  
- **GPU not used**: if no CUDA, runs on CPU.  

---

## 📚 References

- [TrafficSign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/TrafficSign-german-traffic-sign)  
- Activity 2 CNN architecture and best practices  
- [OpenCV Python Docs](https://docs.opencv.org/)

---

## 👨‍💻 Author
Sudhir Kumar Bera – [GitHub](https://github.com/sberaconnects)

