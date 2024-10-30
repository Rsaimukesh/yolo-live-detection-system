# Real-Time Object Detection using YOLOv8 and OpenCV

This project implements real-time object detection using the YOLOv8 model from the Ultralytics library. It uses OpenCV for video capture and rendering. The script detects objects from the camera feed, draws bounding boxes around detected objects, and shows the FPS and object counts on the frame. It also allows toggling detection on and off, and saves frames with detected objects.

## Features
- **Real-time object detection** using YOLOv8 with GPU support (if available)
- **Frame-by-frame FPS calculation** and display
- **Object count per class**, displayed on the frame
- **Toggle detection on/off** during runtime
- **Save frames with detections** to a specified directory

## Requirements
1. **Python 3.8+**
2. **Required Libraries**
   - OpenCV: `pip install opencv-python`
   - PyTorch: `pip install torch`
   - Ultralytics: `pip install ultralytics`
   - NumPy: `pip install numpy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download YOLOv8 model weights and place them in a `weights/` directory:
   ```plaintext
   weights/yolov8n.pt
   ```

## Usage
1. Run the script:
   ```bash
   python object_detection.py
   ```
2. Select the camera index (0 for default camera, 1 for an external camera, etc.).

3. Use the following key commands:
   - **Press `q`** to quit.
   - **Press `d`** to toggle detection on/off.

## File Structure
- **object_detection.py**: Main script for object detection
- **weights/yolov8n.pt**: YOLOv8 weights
- **utils/coco.txt**: Class names for the COCO dataset
- **detections/**: Folder to save frames with detected objects

## Code Overview
- **select_camera()**: Allows user to select a camera based on index input.
- **YOLO Model Loading**: Loads a YOLOv8 model with GPU support (if available).
- **Frame Processing**: Captures frames, performs detection, calculates FPS, and renders detected objects.
- **Object Counting**: Counts each detected object per class and displays it on the frame.
- **Frame Saving**: Saves frames with detections to the `detections/` directory.

## Example Output
![Example Output](example_output.jpg)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
