import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43

# Centralized Paths
MODEL_PATH = os.path.join('..', 'model',
                          'traffic_sign', 'best_traffic_sign_model.keras')
TEST_DATA_PATH = os.path.join('..', 'data', 'TrafficSign', 'Test')

# Load model
model = load_model(MODEL_PATH)

# Load test data


def load_test_data(test_path):
    images = []
    labels = []
    class_folders = sorted(os.listdir(test_path))

    for class_id in class_folders:
        class_folder = os.path.join(test_path, class_id)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            try:
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(int(class_id))
            except:
                continue

    return np.array(images), np.array(labels)


print("Loading test data...")
x_test, y_test = load_test_data(TEST_DATA_PATH)

if len(x_test) == 0 or len(y_test) == 0:
    raise ValueError(
        "\u274c No test data found. Please check your TEST_DATA_PATH or dataset structure.")

x_test = x_test.astype('float32') / 255.0
print(f"Loaded {len(x_test)} test images across {len(set(y_test))} classes")

# Predict
print("Running predictions...")
pred_probs = model.predict(x_test)
y_pred = np.argmax(pred_probs, axis=1)

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as 'confusion_matrix.png'")
