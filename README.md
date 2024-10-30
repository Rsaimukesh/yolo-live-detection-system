Real-Time Object Detection using YOLOv8 and OpenCV
This project implements real-time object detection using the YOLOv8 model from the Ultralytics library. It uses OpenCV for video capture and rendering. The script detects objects from the camera feed, draws bounding boxes around detected objects, and shows the FPS and object counts on the frame. It also allows toggling detection on and off, and saves frames with detected objects.

Features
Real-time object detection using YOLOv8 with GPU support (if available)
Frame-by-frame FPS calculation and display
Object count per class, displayed on the frame
Toggle detection on/off during runtime
Save frames with detections to a specified directory
Requirements
Python 3.8+
Required Libraries
OpenCV: pip install opencv-python
PyTorch: pip install torch
Ultralytics: pip install ultralytics
NumPy: pip install numpy
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Download YOLOv8 model weights and place them in a weights/ directory:
plaintext
Copy code
weights/yolov8n.pt
Usage
Run the script:

bash
Copy code
python object_detection.py
Select the camera index (0 for default camera, 1 for an external camera, etc.).

Use the following key commands:

Press q to quit.
Press d to toggle detection on/off.
File Structure
object_detection.py: Main script for object detection
weights/yolov8n.pt: YOLOv8 weights
utils/coco.txt: Class names for the COCO dataset
detections/: Folder to save frames with detected objects
Code Overview
select_camera(): Allows user to select a camera based on index input.
YOLO Model Loading: Loads a YOLOv8 model with GPU support (if available).
Frame Processing: Captures frames, performs detection, calculates FPS, and renders detected objects.
Object Counting: Counts each detected object per class and displays it on the frame.
Frame Saving: Saves frames with detections to the detections/ directory.
Example Output

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Ultralytics YOLO
OpenCV
