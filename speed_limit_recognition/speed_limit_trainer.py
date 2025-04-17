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

# Centralized paths
DATA_DIR = os.path.join("..", "data", "SpeedLimit")
MODEL_DIR = os.path.join("..", "model", "speed_limit")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 8  # classes 0 through 7

# Load images and labels


def load_data(split):
    path = os.path.join(DATA_DIR, split)
    images, labels = [], []
    for cls in range(NUM_CLASSES):
        cls_path = os.path.join(path, str(cls))
        for fname in os.listdir(cls_path):
            img = cv2.imread(os.path.join(cls_path, fname))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(cls)
    return np.array(images), np.array(labels)


# Prepare data
print("Loading train data...")
x_train, y_train = load_data('Train')
print("Loading test (for validation) data...")
x_test, y_test = load_data('Test')

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Encode labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Split train into train/val
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
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
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for best model saving
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_speed_limit_model.keras'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)

# Train
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint]
)

# Save final model
model.save(os.path.join(MODEL_DIR, 'speed_limit_model.keras'))
print("Models saved to", MODEL_DIR)

# Plot and save accuracy curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Speed Limit Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('training_accuracy_speed_limit.png')
