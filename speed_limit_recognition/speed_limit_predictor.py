import os
import cv2
import numpy as np
import tensorflow as tf

# Centralized paths
model_path = os.path.join('..', 'model',
                          'speed_limit', 'best_speed_limit_model.keras')
# Class labels for speed limits (20km/h to 120km/h)
class_labels = ["20 km/h", "30 km/h", "50 km/h", "60 km/h",
                "70 km/h", "80 km/h", "100 km/h", "120 km/h"]

# Load model
model = tf.keras.models.load_model(model_path)

# Prediction function


def predict_speed_limit(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (32, 32))
    img_norm = img_resized.astype('float32') / 255.0
    inp = np.expand_dims(img_norm, axis=0)

    probs = model.predict(inp)
    cls = np.argmax(probs)
    conf = probs[0][cls]
    label = class_labels[cls]

    print(f"✅ Predicted speed limit: {label} | Confidence: {conf:.2f}")

    # Show result
    result = cv2.putText(img.copy(), f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                         0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Speed Limit Prediction', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = input("Enter path to speed limit image: ")
    predict_speed_limit(path)
