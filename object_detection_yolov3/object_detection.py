import cv2
import os
import argparse
import numpy as np

# Centralized model and data paths
MODEL_DIR = os.path.join("..", "model", "yolov3")
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Load YOLO files
yolo_cfg = os.path.join(MODEL_DIR, "yolov3.cfg")
yolo_weights = os.path.join(MODEL_DIR, "yolov3.weights")
yolo_classes = os.path.join(MODEL_DIR, "coco.names")

# Load classes
with open(yolo_classes, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Filter for road-relevant objects
filter_classes = ['car', 'bus', 'person', 'bicycle',
                  'traffic light', 'stop sign', 'street lamp']

# Set colors
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load YOLO
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1]
              for i in net.getUnconnectedOutLayers().flatten()]


def detect_objects(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                label = class_names[class_id]
                if label in filter_classes:
                    box = detection[0:4] * \
                        np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")

                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply NMS
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = colors[class_ids[i]]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # Show result
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs='+', required=True,
                        help="Path(s) to input image(s)")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--nms-thres", type=float, default=0.4)
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf_thres
    NMS_THRESHOLD = args.nms_thres

    for img_path in args.images:
        if os.path.exists(img_path):
            detect_objects(img_path)
        else:
            print(f"❌ Image not found: {img_path}")
