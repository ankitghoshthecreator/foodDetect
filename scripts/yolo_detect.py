import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', default='D:/yolo_project_python39/models/my_model.onnx')
parser.add_argument('--source', help='Path to image file (leave empty for webcam)')
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
args = parser.parse_args()

# Load YOLO model
yolo_model = YOLO(args.model)


def detect_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        sys.exit()

    results = yolo_model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf >= args.thresh:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result.names[box.cls[0].item()]}: {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Image Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_real_time_detection():
    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                if conf >= args.thresh:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{result.names[box.cls[0].item()]}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if args.source:
        detect_image(args.source)
    else:
        run_real_time_detection()