from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 detector.
        Args:
            model_path (str): Path to the YOLOv8 model weights.
        """
        self.model = YOLO(model_path)

    def get_class_names(self):
        """
        Returns the class names of the model.
        """
        return self.model.names

    def detect(self, image: np.ndarray):
        """
        Performs object detection on the given image.
        Args:
            image (np.ndarray): The input image in OpenCV format (BGR).
        Returns:
            list: A list of dictionaries, each containing 'box' (xyxy format) and 'conf' (confidence).
        """
        results = self.model(image)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf
                })
        return detections

if __name__ == '__main__':
    # Example usage:
    detector = YOLODetector()
    # Create a dummy image (e.g., a black image)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    # You would typically load an image like this:
    # image_path = "path/to/your/image.jpg"
    # image = cv2.imread(image_path)
    
    detections = detector.detect(dummy_image)
    print(f"Detected objects: {detections}")
