import yolov5
import cv2
import torch
import numpy as np

def filter_predictions(results):
  predictions = results.pred[0]
  categories = predictions[:, 5]
  human_idx = (categories == 0).nonzero()
  predictions = results.pred[0][human_idx]
  predictions = predictions[human_idx]
  boxes = predictions[:, :4]
  boxes = boxes[human_idx]
  return torch.squeeze(predictions), torch.squeeze(boxes)

def draw_predictions(frame, boxes):
  boxes = boxes.numpy().astype(int)
  if np.ndim(boxes) < 2:
    boxes = np.reshape(boxes, [1,-1])
  for i in range(boxes.shape[0]):
    cv2.rectangle(frame, (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), (0, 255, 0), 2)
  return frame


model = yolov5.load('yolov5s.pt')
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 30  # maximum number of detections per image

cap = cv2.VideoCapture('videos/test.mp4')
while(cap.isOpened()):
  ret, frame = cap.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = model(frame)
  prediction, boxes = filter_predictions(results)
  frame = draw_predictions(frame, boxes)
  cv2.imshow('frame', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break




# predictions = results.pred[0]
# boxes = predictions[:, :4]  # x1, y1, x2, y2
# scores = predictions[:, 4]
# categories = predictions[:, 5]

# show detection bounding boxes on image
