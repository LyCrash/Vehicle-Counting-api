# ESP32Cam-ObjectCounting

## Project description
This project aims to create a real-time object counting application using an ESP32-CAM module to capture images and a server (PC) for object detection using the YOLOv3/YOLOv7 models with OpenCV and ikomia in Python. 
- **main branch**: All objects listed in the coco.names file are detected and the total count is displayed.


## Project Components:
1. ESP32-CAM Code

    The ESP32-CAM captures images at different resolutions (low, medium, high) and sends them to the server.
    Images are accessible through HTTP endpoints (/cam-lo.jpg, /cam-hi.jpg, /cam-mid.jpg).

2. Server Code

    The server retrieves images from the ESP32-CAM and performs object detection on these images, specifically cars, trucks and buses.
    Many approaches are performed for the counting that is displayed in real-time on the server's graphical user interface.

## Requirements:

    - ESP32-CAM module (AI Thinker model) with appropriate connections, I suggest you to test the camera using `test_live_cam.py`
    - Python environment installed on the server (run `pip install -r requirements.txt`)
    - Download the YOLOv3-320 weights and cfg files from darknet [website](https://pjreddie.com/darknet/yolo/).

## Setup:

    - Configure ESP32-CAM code with appropriate Wi-Fi credentials in the file `capture-esp32cam.ino`
    - Upload the ESP32-CAM code on the module using Arduino IDE. 
    - Ensure the server has all requirements installed.
    - Run the server code to start object detection/counting.

## Usage:

    Access the ESP32-CAM images through HTTP endpoints provided by the server.
    Real-time object counts will be displayed on the server's graphical user interface.

Feel free to customize the project for different objects or resolutions based on your specific requirements.
