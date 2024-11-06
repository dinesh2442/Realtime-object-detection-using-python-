import cv2
import numpy as np
from typing import List, Tuple
import tkinter as tk
from tkinter import PhotoImage
import signal
import sys
import pygame
import time

# Initialize Pygame for sound
pygame.init()
sound = pygame.mixer.Sound("alarm1.mp3")  # Ensure this sound file is in the same directory


class ObjectDetector:
    """
    A class for real-time object detection using YOLO, with distance measurement and sound/image capture capability.
    """

    def __init__(self, weights_path: str, config_path: str, classes_path: str, focal_length: float,
                 known_object_height: float, known_object_width: float, use_gpu: bool = True):
        """
        Initialize the ObjectDetector with model paths and distance calculation parameters.


        """
        self.net = cv2.dnn.readNet(weights_path, config_path)
        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("[INFO] GPU is available. Using CUDA backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("[INFO] GPU not found or not supported. Using CPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.focal_length = focal_length
        self.known_object_height = known_object_height
        self.known_object_width = known_object_width

    def detect_objects(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int], float]]:
        """
        Detect objects in a single frame and calculate their distances.

        :param frame: Input frame as a numpy array
        :return: List of tuples containing (class_name, confidence, bounding_box, distance)
        """
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                if np.any(scores):
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        x, y = int(center_x - w / 2), int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        results = []
        for i in indices:
            box = boxes[i]
            class_name = self.classes[class_ids[i]]
            confidence = confidences[i]
            distance = self.calculate_distance(box[3])  # Using the height of the bounding box
            results.append((class_name, confidence, tuple(box), distance))
        return results

    def calculate_distance(self, perceived_height: int) -> float:
        """
        Calculate the distance to an object based on its perceived height in the image.

        :param perceived_height: The height of the bounding box in pixels
        :return: Estimated distance to the object
        """
        if perceived_height == 0:
            return float('inf')  # Avoid division by zero
        return (self.known_object_height * self.focal_length) / perceived_height

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Tuple[str, float, Tuple[int, int, int, int], float]]) -> np.ndarray:
        """
        Draw bounding boxes, labels, and distances on the frame.

        :param frame: Input frame
        :param detections: List of detections from detect_objects method
        :return: Frame with drawn detections
        """
        for class_name, confidence, (x, y, w, h), distance in detections:
            color = self.colors[self.classes.index(class_name)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}: {confidence:.2f}, Dist: {distance:.2f} cm"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


class ObjectDetectionApp:
    """
    A class to manage the object detection application.
    """

    def __init__(self, weights_path: str, config_path: str, classes_path: str, focal_length: float,
                 known_object_height: float, known_object_width: float, use_gpu: bool = True):
        """
        Initialize the ObjectDetectionApp.

        :param weights_path: Path to the YOLO weights file
        :param config_path: Path to the YOLO configuration file
        :param classes_path: Path to the file containing class names
        :param focal_length: The calibrated focal length of the camera
        :param known_object_height: The known height of the object in real-world units (e.g., centimeters)
        :param known_object_width: The known width of the object in real-world units (e.g., centimeters)
        :param use_gpu: Whether to use GPU acceleration
        """
        self.detector = ObjectDetector(weights_path, config_path, classes_path, focal_length, known_object_height,
                                       known_object_width, use_gpu)
        self.cap = cv2.VideoCapture(0)
        self.is_running = False

    def start_detection(self):
        """
        Start the object detection application.
        """
        self.is_running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect_objects(frame)
            person_detected = any(d[0] == "person" for d in detections)

            if person_detected:
                sound.play()  # Play sound if a person is detected

                # Capture the frame as an image when a person is detected
                timestamp = int(time.time())
                cv2.imwrite(f"person_detection_{timestamp}.jpg", frame)
                print(f"Image saved as person_detection_{timestamp}.jpg")

            frame_with_detections = self.detector.draw_detections(frame, detections)
            cv2.imshow("Real-time Object Detection", frame_with_detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        """
        Stop the object detection application and release resources.
        """
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application stopped. Resources released.")
        sys.exit(0)

    def signal_handler(self, sig, frame):
        """
        Handle interruption signals.
        """
        print("\nReceived interruption signal. Stopping the application...")
        self.stop()


def display_gui(app):
    """
    Display a Tkinter GUI with a 'Continue' button that starts object detection when clicked.
    """
    root = tk.Tk()
    root.title("Object Detection App")

    # Load a sample image for the initial screen
    initial_img = PhotoImage(file="sample_image.png")  # Ensure this image file is in the same directory

    canvas = tk.Canvas(root, width=initial_img.width(), height=initial_img.height())
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=initial_img)

    # Add "Continue" button to start detection
    continue_button = tk.Button(root, text="Get Started", command=lambda: (root.destroy(), app.start_detection()))
    continue_button.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()


def main():
    # Paths to the pre-trained model and configuration files
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    classes_path = "coco.names"

    # Distance measurement parameters
    focal_length = 615  # Example value, calculate this based on your calibration process
    known_object_height = 8.37  # Example object height in centimeters
    known_object_width = 15.92  # Example object width in centimeters

    app = ObjectDetectionApp(weights_path, config_path, classes_path, focal_length, known_object_height,
                             known_object_width, use_gpu=True)
    display_gui(app)


if __name__ == "__main__":
    main()
