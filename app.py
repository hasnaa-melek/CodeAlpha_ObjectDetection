import streamlit as st
from PIL import Image
import numpy as np
import cv2
from detector import ObjectDetector


st.set_page_config(
    page_title="AI Object Detection",
    page_icon="Object Detection",
    layout="wide"
)

st.title("AI Object Detection")
st.write("YOLOv4-Tiny object detection using OpenCV DNN")


@st.cache_resource
def load_detector():
    return ObjectDetector()


detector = load_detector()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.subheader("Original Image")
    st.image(image_array, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            detections = detector.detect(image_array)

        result = image_array.copy()

        for detection in detections:
            x, y, w, h = detection["box"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                result,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        st.subheader("Detection Result")
        st.image(
            result,
            channels="RGB",
            use_container_width=True
        )

        st.subheader("Detected Objects")

        if detections:
            for detection in detections:
                st.write(detection)
        else:
            st.info("No objects detected.")