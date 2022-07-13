import cv2
from detector_module import HumanDetector
import time
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
import pickle


cap = cv2.VideoCapture('videos/wembley/cam02.mp4')
pTime = 0
kps_thres = 0.7

yolo_det = HumanDetector(img_size=1920)
pose_config = 'configs/hrnet_w48_coco_256x192.py'
pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
pose_model = init_pose_model(pose_config, pose_checkpoint)

detections = []
while True:
    success, img = cap.read()
    imgDraw = img.copy()
    bboxes = yolo_det.predict(img)
    scores = bboxes[:, 4]

    mask = bboxes[:, 4] > 0.5
    person_results = [{'bbox': b[:5]} for b in bboxes[mask, :]]
    detections = [b[:5] for b in bboxes[mask, :]]

    pose_results, returned_outputs = inference_top_down_pose_model(
                                                            pose_model,
                                                            img,
                                                            person_results,
                                                            bbox_thr=0.0,
                                                            format='xyxy',
                                                            dataset=pose_model.cfg.data.test.type)
    poses = [p['keypoints'] for p in pose_results]

    imgDraw = vis_pose_result(
                        pose_model,
                        imgDraw,
                        pose_results,
                        kpt_score_thr=kps_thres,
                        dataset=pose_model.cfg.data.test.type,
                        show=False)

    # Create detections - poses dictionary and save it.
    detections_dict = {'detections': np.array(detections), 'poses': np.array(poses)}
    # with open("detections/detections_dict_cam02.pkl", "wb") as f:
    #     pickle.dump(detections_dict, f)
    #     f.close()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(imgDraw, dim, interpolation=cv2.INTER_AREA)
    cv2.putText(resized, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", resized)
    cv2.waitKey(1)



