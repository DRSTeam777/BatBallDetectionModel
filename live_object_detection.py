import cv2
import torch
from pathlib import Path
import sys
import numpy as np

# Add yolov5 directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Initialize model
weights = 'best.pt'
device = select_device('')
model = attempt_load(weights, device)  # load FP32 model
stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

# Load video
video_path = 'cricket_test.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img0, (416, 416))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference
    pred = model(img, augment=False)[0]

    # Postprocess detections
    pred = non_max_suppression(pred, 0.4, 0.5, agnostic=False)

    # Visualize detections
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
