import yolov5
import cv2
import torch
import numpy as np
import time

class YoloDetector:
    def __init__(self, weights_path='yolov5s.pt', classes=[0], conf=0.25, iou_threshold=0.45,
                 multi_label=False, max_det=30):


        self.model = yolov5.load(weights_path)
        self.model.classes = classes
        self.model.conf = conf  # NMS confidence threshold
        self.model.iou = iou_threshold  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = multi_label  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image

    def predict(self, img):
        results = self.model(img)
        return results.pred[0].cpu().numpy()[:, :4]


    def get_bb_img(self, img, bb, expand_bb=False, margin=5):
        if expand_bb:
            bb[0] -= margin
            bb[1] -= margin
            bb[2] += margin
            bb[2] += margin
        bb=np.squeeze(bb.astype(int))
        return bb, img[bb[1]:bb[3], bb[0]:bb[2],:]



def main():
    device = "cuda:0"  # or "cpu"
    cap = cv2.VideoCapture('videos/test.mp4')  # make VideoCapture(0) for webcam
    pTime = 0
    detector = YoloDetector()
    while True:
        success, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = detector.predict(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
