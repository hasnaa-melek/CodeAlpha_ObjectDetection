import cv2
import numpy as np


MODEL_PATH = "models/yolov4-tiny.onnx"

INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45


class ObjectDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)

        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

    def detect(self, image):
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)

        outputs = self.net.forward(
            self.net.getUnconnectedOutLayersNames()
        )

        boxes = outputs[0]
        scores = outputs[1]

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        detected_boxes = []
        confidences = []
        class_ids = []

        for i in range(len(boxes)):
            class_id = int(np.argmax(scores[i]))
            confidence = float(scores[i][class_id])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            box = boxes[i]

            center_x = float(box[0]) * width
            center_y = float(box[1]) * height
            box_width = float(box[2]) * width
            box_height = float(box[3]) * height

            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            detected_boxes.append([x, y, int(box_width), int(box_height)])
            confidences.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            detected_boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        detections = []

        if len(indices) > 0:
            for index in np.array(indices).flatten():
                x, y, w, h = detected_boxes[index]

                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                class_id = class_ids[index]

                detections.append({
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": confidences[index],
                    "box": [x, y, w, h]
                })

        return detections


def draw_detections(image, detections):
    result = image.copy()

    for detection in detections:
        x, y, w, h = detection["box"]
        label = detection["class_name"]
        confidence = detection["confidence"]

        text = f"{label}: {confidence:.2f}"

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        text_y = max(y - 10, 20)

        cv2.putText(
            result,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return result