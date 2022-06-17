import cv2
from detector_module import YoloDetector
from pose_module import PoseDetector
import time
import mediapipe as mp


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture('videos/wembley/cam01.mp4')
pTime = 0

yolo_det = YoloDetector()
pose_det = PoseDetector()

while True:
    success, img = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = yolo_det.predict(img)
    poses = []
    for i in range(bboxes.shape[0]):
        bboxes[i,:], croped_img = yolo_det.get_bb_img(img, bboxes[i,:], expand_bb=True, margin=5)
        poses.append(pose_det.predict(croped_img))
        # print(results.pose_landmarks)
        h, w, c = croped_img.shape
        for j in range(poses[i].shape[0]):
            cx, cy = int(poses[i][j,0] * w), int(poses[i][j,0] * h)
            croped_img=cv2.circle(croped_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.imshow("Image", croped_img)
        cv2.waitKey(1)