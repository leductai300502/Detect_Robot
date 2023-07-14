from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import numpy
import numpy as np
from imutils.video import VideoStream
import time

model = YOLO('yolov8n.pt')
# model.info()  # display model information
# points = []





results = model.predict(source = "0", show =True ,save_txt = True  )



