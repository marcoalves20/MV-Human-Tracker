from yolov5 import YOLOv5
import cv2
import torch
import numpy as np
import time

class YoloDetector:
    def __init__(self, weights_path, device='cpu', classes=[0], min_detection_confidence=0.5, iou_threshold=1,
                 multi_label=False, max_det=False):
        self.model = YOLOv5(weights_path, device=device)
        self.model.classes = classes
        self.model.conf = min_detection_confidence  # NMS confidence threshold
        self.model.iou = iou_threshold  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = multi_label  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image

    def run_inference(self, img):
        results = self.model.predict(img)
        bboxes = []
        if results:
            for result in results.pred:
                bboxes.append(result.cpu().numpy()[:,:4])
            return bboxes
        else:
            return []



def main():
    device = "cuda:0"  # or "cpu"
    cap = cv2.VideoCapture('videos/wembley/cam01.mp4')  # make VideoCapture(0) for webcam
    pTime = 0
    detector = YoloDetector('models/yolov5s.pt', device)
    while True:
        success, img = cap.read()
        bboxes = detector.run_inference(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
