import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# Centralized paths
data_dir = os.path.join('..', 'data', 'SpeedLimit', 'Test')
model_path = os.path.join('..', 'model',
                          'speed_limit', 'best_speed_limit_model.keras')

# Load model
model = tf.keras.models.load_model(model_path)

# Load test data


def load_test_data(path):
    images, labels = [], []
    for cls in sorted(os.listdir(path), key=int):
        cls_folder = os.path.join(path, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            img = cv2.imread(os.path.join(cls_folder, fname))
            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(int(cls))
    return np.array(images), np.array(labels)


print("Loading test images...")
x_test, y_true = load_test_data(data_dir)
if len(x_test) == 0:
    raise ValueError("No test data found in {}".format(data_dir))

# Preprocess
x_test = x_test.astype('float32') / 255.0

# Predict
print("Predicting...")
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Speed Limit Recognition Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_speed_limit.png')
print("\nConfusion matrix saved as 'confusion_matrix_speed_limit.png'")
