import streamlit as st
import cv2
from object_detection import ObjectDetector
import numpy as np
import tempfile
from PIL import Image

# Initialize the object detector
MODEL_PATH = 'yolo-weights/yolov8n.pt'
detector = ObjectDetector(MODEL_PATH)

# Streamlit UI elements
st.title("Object Detection with YOLO")
st.write("Upload an image or video file, or use the webcam for object detection.")

# Option to upload image, video file, or use webcam
option = st.selectbox("Select an option:", ["Upload Image", "Upload Video", "Use Webcam"])

if option == "Upload Image":
    # Image upload
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Read and process the image
        image = Image.open(uploaded_image)
        img = np.array(image)

        # Detect objects
        detections = detector.detect_objects(img)

        # Draw detections on the image
        detector.draw_detections(img, detections)

        # Display the image
        st.image(img, channels="RGB", use_column_width=True)

elif option == "Upload Video":
    # Video upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

    if uploaded_file:
        # Save the uploaded video file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        # Streamlit video display
        stframe = st.empty()

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Detect objects
            detections = detector.detect_objects(img)

            # Draw detections on the image
            detector.draw_detections(img, detections)

            # Convert image to RGB format and display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stframe.image(img_rgb, channels="RGB", use_column_width=True)

        cap.release()

elif option == "Use Webcam":
    # Webcam feed
    st.write("Webcam feed:")
    run_webcam = st.checkbox("Start Webcam")
    
    cap = cv2.VideoCapture(0)  # for WebCam
    cap.set(1, 1280)  # width
    cap.set(2, 920)  # height
        
    # Streamlit video display
    stframe = st.empty()

    # Button to stop webcam feed
    stop_button = st.button("Stop Webcam", key='stop_button_camera')

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image from camera.")
            break

        # Detect objects
        detections = detector.detect_objects(frame)

        # Draw detections on the image
        detector.draw_detections(frame, detections)
        
        # Convert image to RGB format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)

        # Check if the stop button is pressed
        if stop_button:
            st.write("Processing stopped")
            break

    cap.release()


st.write("**Note:** Use the dropdown menu to choose between image upload, video upload, or webcam feed.")
