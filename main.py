


import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())

import cv2
from ultralytics import YOLO



model = YOLO('Yolo_Weights/yolov8n.pt')
results = model("Images/people.jpeg", show=True)

cv2.waitKey(0)