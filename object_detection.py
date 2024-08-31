import math
import cvzone
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"]

    def load_model(self, model_path):
        try:
            model = YOLO(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_objects(self, img):
        try:
            results = self.model(img, stream=True)
            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = (math.ceil(box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    detections.append({
                        "bbox": (x1, y1, w, h),
                        "confidence": conf,
                        "class_name": self.class_names[cls]
                    })

            return detections
        except Exception as e:
            print(f"Error during inference: {e}")
            return []

    def draw_detections(self, img, detections):
        for detection in detections:
            x1, y1, w, h = detection["bbox"]
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{detection["class_name"]} {detection["confidence"]}', 
                               (max(0, x1), max(35, y1)), scale=2)

