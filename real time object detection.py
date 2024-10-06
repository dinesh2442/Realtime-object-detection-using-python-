import threading
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from threading import Thread

class ObjectDetection:
    def __init__(self, num):
        self.capture = cv2.VideoCapture(0)  # opens laptop's video cam

        self.yolo = cv2.dnn.readNet("./yolov3.cfg", "./yolov3.weights")

        self.classes = []  # 80 classes
        with open("./coco.names", "r") as f:
            self.classes = f.read().splitlines()

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        try:
            if self.status:
                img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
                self.yolo.setInput(blob)

                output_layer_names = self.yolo.getUnconnectedOutLayersNames()
                layer_output = self.yolo.forward(output_layer_names)

                boxes = []
                confidences = []
                class_ids = []
                h, w = img.shape[:2]
                for output in layer_output:
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if confidence > 0.4:
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            box = [x, y, int(width), int(height)]
                            boxes.append(box)
                            confidences.append(float(confidence))
                            class_ids.append(classID)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                if len(boxes) != 0:
                    for i in indexes.flatten():
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        conf = str(round(confidences[i], 2))
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 10)
                        cv2.putText(img, label + " " + conf, (x, y + 20), font, 2, (255, 255, 255), 2)

                    img1 = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img1.resize((800, 800)))  # increased size
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    lmain.after(0, self.show_frame())  # this method is running infinitely
                else:
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img1 = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img1.resize((800, 800)))  # increased size
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    lmain.after(2, self.show_frame())

        except tk.TclError:
            print(" ")


def cam(number):
    rtsp_stream_link = number
    video_stream_widget = ObjectDetection(number)
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass


def getwebcam():
    global newWindow, lmain
    newWindow = tk.Toplevel(root)
    newWindow.title("Live Detection")

    newWindow.geometry("800x800")  # increased size
    lmain = tk.Label(newWindow, text='Starting....................')
    lmain.place(x=0, y=0)
    t1 = threading.Thread(target=cam, args=(0,))
    t1.start()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("750x394")
    root.title("Object Detection")
    root.configure(bg="black")

    width = 750
    height = 394
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry("%dx%d+%d+%d" % (width, height, x, y))

    # create a label widget for the background image
    bg_image = tk.PhotoImage(file="./tkBackground.png")
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # create the button for live detection
    button3 = tk.Button(root, text="Live Detection", font=("Helvetica", 16), command=getwebcam, bd="5")
    button3.place(x=300, y=290)

    root.mainloop()
