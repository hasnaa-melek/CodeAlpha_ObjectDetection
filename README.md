# CodeAlpha Object Detection and Tracking

An AI-powered object detection and tracking application built with Python, OpenCV DNN, YOLOv4-Tiny, NumPy, and Streamlit.

## Project Overview

This project detects objects in images and videos using a pre-trained YOLOv4-Tiny ONNX model. The application provides a Streamlit interface where users can upload an image for object detection or a video for object detection and tracking.

For images, the application displays detected objects with bounding boxes, class names, and confidence scores.

For videos, detected objects are tracked across consecutive frames using a lightweight IoU-based tracking algorithm. Each tracked object receives a unique tracking ID.

## Features

- Object detection using YOLOv4-Tiny
- OpenCV DNN inference
- ONNX model support
- Image upload through Streamlit
- Video upload through Streamlit
- Bounding boxes around detected objects
- Object class labels
- Confidence scores
- Object tracking across video frames
- Unique tracking IDs
- Processed video output
- Detection and tracking results displayed directly in the web application
- Lightweight implementation without PyTorch or Ultralytics

## Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit
- YOLOv4-Tiny
- ONNX

## Project Structure

```text
CodeAlpha_ObjectDetection/
│
├── app.py
├── detector.py
├── tracker.py
├── test_model.py
├── requirements.txt
│
├── models/
│   └── yolov4-tiny.onnx
│
└── output/
