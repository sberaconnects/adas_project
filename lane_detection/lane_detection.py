# Lane Detection System using OpenCV
# Initial Setup for Frame Processing and Edge Detection

import cv2
import numpy as np

# Load video (replace with your downloaded KITTI video file path)
cap = cv2.VideoCapture('kitti_road_video.mp4')

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = None

# Define region of interest vertices (adjust based on video resolution)
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([
        [(0, height), (width, height), (width, int(height*0.6)), (0, int(height*0.6))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Draw detected lines on the original frame
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cv2.addWeighted(img, 0.8, line_image, 1, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)

    # Line detection
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # Draw lines on the original frame
    line_image = display_lines(frame, lines)

    # Initialize video writer if not set
    if out is None:
        height, width = frame.shape[:2]
        out = cv2.VideoWriter('lane_output.mp4', fourcc, 20.0, (width, height))

    out.write(line_image)
    cv2.imshow('Lane Detection', line_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

