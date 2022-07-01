import cv2
from detector_module import HumanDetector
from pose_module import PoseDetector
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


def debug_plot(croped_img, poses):
    plt.imshow(croped_img)
    plt.scatter(poses[0][:,0],poses[0][:,1], s=2)
    plt.show()


cap = cv2.VideoCapture('videos/wembley/cam02.mp4')
pTime = 0
kps_thres = 0.7

yolo_det = HumanDetector(img_size=1920)
pose_det = PoseDetector(static_image_mode=True, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.4)
detections = []
while True:
    success, img = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = yolo_det.predict(img)
    scores = bboxes[:, 4]

    poses = []
    for i in range(bboxes.shape[0]):
        # if bbox score is smaller than a threshold ignore it
        if scores[i] < 0.5:
            break

        detections.append(bboxes[i, :])
        cropped_img = yolo_det.get_bb_img(img, bboxes[i, :], expand_bb=True, margin=5)
        poses.append(pose_det.predict(cropped_img))

        if np.all(poses[i] != 0):
            for j in range(poses[i].shape[0]):
                cx, cy = int(poses[i][j,0] ), int(poses[i][j,1] )
                x, y = int(poses[i][j,0] + bboxes[i][0]), int(poses[i][j,1] + bboxes[i][1])
                if poses[i][j, 2] > kps_thres:
                    cv2.circle(cropped_img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, (x, y), 3, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(cropped_img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x, y), 3, (0, 0, 255), cv2.FILLED)

        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(1)

    # Create detections - poses dictionary and save it.
    detections_dict = {'detections': np.array(detections), 'poses': np.array(poses)}
    # with open("detections/detections_dict_cam02.pkl", "wb") as f:
    #     pickle.dump(detections_dict, f)
    #     f.close()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # resize image
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.putText(resized, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", resized)
    cv2.waitKey(1)


