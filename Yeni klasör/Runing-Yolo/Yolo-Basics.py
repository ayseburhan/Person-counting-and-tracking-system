from ultralytics import YOLO
import cv2
from cv2 import waitKey

model=YOLO("Yolo-Weights/yolov8l.pt")
results=model("Images/3.jpeg", show=True)
waitKey(0)