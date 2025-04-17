import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Centralized Paths
MODEL_PATH = os.path.join('..', 'model', 'traffic_sign',
                          'best_traffic_sign_model.keras')

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# TrafficSign official class labels
class_labels = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

# Prediction function


def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image path does not exist.")
        return

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (32, 32))
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_input)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    label = class_labels[class_id]

    print(
        f"✅ Predicted: {label} (Class {class_id}) | Confidence: {confidence:.2f}")

    # Display with label
    img_bgr = cv2.putText(img.copy(), label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                          0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Prediction", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    test_image_path = input("Enter image path to predict: ")
    predict_image(test_image_path)
