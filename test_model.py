import cv2
import numpy as np

MODEL_PATH = "models/yolov4-tiny.onnx"

print("Loading YOLOv4-Tiny ONNX model...")

net = cv2.dnn.readNetFromONNX(MODEL_PATH)

print("Model loaded successfully.")

print("\nModel input information:")
for layer in net.getLayerNames()[:10]:
    print(layer)

print("\nTesting model with a blank image...")

image = np.zeros((416, 416, 3), dtype=np.uint8)

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1 / 255.0,
    size=(416, 416),
    swapRB=True,
    crop=False
)

net.setInput(blob)

outputs = net.forward(net.getUnconnectedOutLayersNames())

print("\nNumber of output layers:", len(outputs))

for i, output in enumerate(outputs):
    print(f"Output {i + 1} shape:", output.shape)

print("\nYOLOv4-Tiny ONNX model test completed.")