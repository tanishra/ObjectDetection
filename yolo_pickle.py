import pickle
from object_detection import ObjectDetector

# Initialize the object detector with the model path
MODEL_PATH = 'yolo-weights/yolov8n.pt'
detector = ObjectDetector(MODEL_PATH)

# Save the detector object to a pickle file
with open('yolo_detector.pkl', 'wb') as f:
    pickle.dump(detector, f)

print("YOLO model has been pickled successfully.")
