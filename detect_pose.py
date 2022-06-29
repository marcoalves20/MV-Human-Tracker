import cv2
from detector_module import HumanDetector
from pose_module import PoseDetector
import time
import mediapipe as mp

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def debug_plot(croped_img, poses):
    plt.imshow(croped_img)
    plt.scatter(poses[0][:,0],poses[0][:,1], s=2)
    plt.show()


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture('videos/wembley/cam01.mp4')
pTime = 0

yolo_det = HumanDetector()
pose_det = PoseDetector()

while True:
    success, img = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = yolo_det.predict(img)

    poses = []
    for i in range(bboxes.shape[0]):
        bboxes[i,:], cropped_img = yolo_det.get_bb_img(img, bboxes[i,:], expand_bb=True, margin=5)
        poses.append(pose_det.predict(cropped_img))
        h, w, c = cropped_img.shape
        for j in range(poses[i].shape[0]):
            cx, cy = int(poses[i][j,0] ), int(poses[i][j,1] )
            cropped_img=cv2.circle(cropped_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.imshow("Image", cropped_img)
        cv2.waitKey(1)
        print(i)


