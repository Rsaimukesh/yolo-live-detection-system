import random
import cv2
import numpy as np
import torch  # Import torch to check for CUDA availability
import time  # For calculating FPS
import os  # For saving frames
from ultralytics import YOLO

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to select the camera based on user input
def select_camera():
    cam_index = int(input("Enter camera index (0 for default camera, 1 for external, etc.): "))
    return cam_index

# Ask the user for the camera index
camera_index = select_camera()

# Open the file containing class names (COCO dataset classes)
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for each class in the class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model and set it to run on the GPU if available
model = YOLO("weights/yolov8n.pt", "v8").to(device)

# Create an output directory to save detected frames (if it doesn't exist)
output_dir = "detections"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the selected camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Cannot open camera with index {camera_index}")
    exit()

# Variables for additional features
detection_enabled = True  # To toggle detection on/off
frame_count = 0  # To save detected frames
fps_start_time = time.time()  # Initialize start time for FPS calculation

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps_start_time = fps_end_time  # Reset start time for next iteration

    # Avoid division by zero
    if time_diff > 0:
        fps = int(1 / time_diff)
    else:
        fps = 0

    # Predict only if detection is enabled
    if detection_enabled:
        # Predict on the frame using YOLOv8, moving the frame to GPU if CUDA is available
        detect_params = model.predict(source=[frame], conf=0.45, save=False, device=device)

        # Check if there are any detections
        if len(detect_params[0]) != 0:
            object_count = {}  # Dictionary to count objects per class

            for i in range(len(detect_params[0])):

                # Extract the bounding boxes and detection information
                boxes = detect_params[0].boxes
                box = boxes[i]  # Get the ith box
                clsID = int(box.cls.numpy()[0])  # Class ID
                conf = box.conf.numpy()[0]  # Confidence score
                bb = box.xyxy.numpy()[0]  # Bounding box coordinates (x1, y1, x2, y2)

                # Count the objects per class
                class_name = class_list[clsID]
                object_count[class_name] = object_count.get(class_name, 0) + 1

                # Draw a rectangle around the detected object
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),  # Top-left corner
                    (int(bb[2]), int(bb[3])),  # Bottom-right corner
                    detection_colors[clsID],  # Color for the bounding box
                    3  # Thickness of the box
                )

                # Display the class name and confidence score on the frame
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    f"{class_name} {round(conf * 100, 2)}%",  # Class name and confidence
                    (int(bb[0]), int(bb[1]) - 10),  # Position of text (above the bounding box)
                    font,
                    1,
                    (255, 255, 255),  # White text color
                    2,
                )

            # Display the object count on the frame
            y_offset = 30
            for obj, count in object_count.items():
                cv2.putText(frame, f"{obj}: {count}", (10, y_offset), font, 1, (0, 255, 0), 2)
                y_offset += 30

    # Display the calculated FPS on the frame
    cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the resulting frame in full size
    cv2.imshow("Real-Time Object Detection", frame)

    # Save the frame with detections to the output directory
    if detection_enabled:
        frame_count += 1
        output_frame_path = os.path.join(output_dir, f"detected_frame_{frame_count}.jpg")
        cv2.imwrite(output_frame_path, frame)

    # Keypress handling for additional functionality
    key = cv2.waitKey(1)

    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break
    # Toggle detection on/off with 'd'
    if key == ord('d'):
        detection_enabled = not detection_enabled
        print(f"Detection {'enabled' if detection_enabled else 'disabled'}.")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows() 