import streamlit as st
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
import math
from PIL import Image
import tempfile

# Load the YOLO model
model = YOLO('yolo-weights/yolov8n.pt')

# List of class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Function to process the image and perform object detection
def detect_objects(frame):
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = (math.ceil(box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.cornerRect(frame, (x1, y1, w, h))
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2)
    return frame

# Streamlit UI
st.title("Object Detection System")

# File uploader for video files
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], key='video_uploader')
use_camera = st.checkbox('Use Webcam for Real-Time Detection', key='camera_checkbox')

# File uploader for image files
uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], key='image_uploader')

# Streamlit placeholder for video and image streams
stframe = st.empty()

if uploaded_image:
    # Process the uploaded image
    image = Image.open(uploaded_image)
    frame = np.array(image)
    frame = detect_objects(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(rgb_frame, channels="RGB", use_column_width=True)

elif uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    
    stop_button = st.button("Stop", key='stop_button')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        
        if stop_button:
            st.write("Processing stopped")
            break

    cap.release()

elif use_camera:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Set width
    cap.set(4, 720)   # Set height
    
    stop_button = st.button("Stop", key='stop_button_camera')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        
        if stop_button:
            st.write("Processing stopped")
            break

    cap.release()

else:
    st.text("Please upload a video file, select the camera for real-time detection, or upload an image file.")
