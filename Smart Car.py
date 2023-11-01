from ultralytics import YOLO
from PIL import Image
import cv2


model = YOLO('model-sourc-addres\\best.pt')

results = model.predict(save = True,source="viedo-url")

