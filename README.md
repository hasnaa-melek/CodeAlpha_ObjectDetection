# CodeAlpha Object Detection

An AI-powered object detection application built with Python, OpenCV DNN, YOLOv4-Tiny, and Streamlit.

## Project Overview

This project detects objects in uploaded images using a pre-trained YOLOv4-Tiny ONNX model. The application provides a simple Streamlit interface where users can upload an image and receive an annotated result with detected objects, bounding boxes, class names, and confidence scores.

## Features

- Object detection using YOLOv4-Tiny
- OpenCV DNN inference
- ONNX model support
- Image upload through Streamlit
- Bounding boxes around detected objects
- Object class labels
- Confidence scores
- Detection results displayed directly in the web application
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
├── test_model.py
├── models/
│   └── yolov4-tiny.onnx
├── input/
├── output/
└── README.md
