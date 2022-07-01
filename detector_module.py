import yolov5
import cv2
import torch
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class HumanDetector:
    def __init__(self, weights_path='yolov5x.pt', img_size=640, classes=[0], conf=0.15, iou_threshold=0.45,
                 multi_label=False, max_det=30):
        self.model = yolov5.load(weights_path)
        self.img_size = img_size
        self.model.classes = classes
        self.model.conf = conf  # NMS confidence threshold
        self.model.iou = iou_threshold  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = multi_label  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image

    def predict(self, img):
        results = self.model(img, size=self.img_size)
        results = results.pred[0].cpu().numpy()
        results = results[results[:,-1] == 0]
        return results


    def get_bb_img(self, img, bb, expand_bb=False, margin=5):
        if expand_bb:
            bb[0] -= margin
            bb[1] -= margin
            bb[2] += margin
            bb[3] += margin
        bb=np.squeeze(bb.astype(int))
        return img[bb[1]:bb[3], bb[0]:bb[2],:]



def main():
    device = "cuda:0"  # or "cpu"
    cap = cv2.VideoCapture('videos/wembley/cam02.mp4')  # make VideoCapture(0) for webcam
    pTime = 0
    detector = HumanDetector(img_size=1920)
    while True:
        success, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = detector.predict(img)
        #bboxes = bboxes.astype(int)
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4].astype(int)
        # np.save('detections_cam05.npy', bboxes)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        for i in range(bboxes.shape[0]):
            cv2.rectangle(img, (bboxes[i,0], bboxes[i,1]), (bboxes[i,2], bboxes[i,3]),(0,255,0),3)
            cv2.putText(img, str(i), (bboxes[i,0] - 10, bboxes[i,1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            cv2.putText(img, str(round(scores[i], 3)), (bboxes[i, 0], bboxes[i, 3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        scale_percent = 50  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.putText(resized, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", resized)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
