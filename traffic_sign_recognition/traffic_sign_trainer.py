import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import cv2

# Centralized Paths
# Centralized paths
DATA_ROOT = os.path.join("..", "data", "TrafficSign")
MODEL_ROOT = os.path.join('..', 'model', 'traffic_sign')
os.makedirs(MODEL_ROOT, exist_ok=True)

# Constants
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43
DATASET_PATH = os.path.join(DATA_ROOT, 'Train')

# Load images and labels


def load_data(dataset_path):
    images = []
    labels = []

    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(dataset_path, str(class_id))
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(class_id)
            except:
                continue

    return np.array(images), np.array(labels)


print("Loading data...")
x_data, y_data = load_data(DATASET_PATH)

# Normalize
x_data = x_data.astype('float32') / 255.0

# Encode labels
lb = LabelBinarizer()
y_data = lb.fit_transform(y_data)

# Split data
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_ROOT, "best_traffic_sign_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint]
)

# Save final model
model.save(os.path.join(MODEL_ROOT, "traffic_sign_cnn_model.keras"))
print("Model saved successfully.")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("training_accuracy.png")
