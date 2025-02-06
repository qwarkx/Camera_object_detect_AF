from ultralytics import YOLO
import cv2
import cvzone

import math


class_names = {  0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
                 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
                 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# cap = cv2.VideoCapture(f'E:/video_anyag - Kepek Mindenes/00 - 2024-05-30 neptanc/Videok/P1033979.MOV')


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model = YOLO('Yolo_Weights/yolov8n.pt')

'''
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    cv2.imshow('image', img)
    cv2.waitKey(1)
'''

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Simple BBOX
            x1, y1, x2, y2 = box.xywh[0]
            x1, y1, w1, h1 = int(x1), int(y1), int(x2), int(y2)
            w, h = w1-x1, h1-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=1)
            cox_conf = box.conf[0]*100
            conf = math.ceil(cox_conf) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{class_names[cls]} {conf}', ( max(0,x1), max(36,y1)), scale=2,thickness=1)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

