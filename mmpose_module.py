import torch, torchvision
import cv2
import mmpose
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
local_runtime = False

print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)
print('mmpose version:', mmpose.__version__)
print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())


JOINT_NAMES_DICT = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

pose_config = 'configs/hrnet_w48_coco_256x192.py'
pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = 'configs/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
det_model = init_detector(det_config, det_checkpoint)

cap = cv2.VideoCapture('videos/cam01.mp4')  # make VideoCapture(0) for webcam

while True:
    success, img = cap.read()
    # inference detection
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    # inference pose
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=0.0,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type)

    # show pose estimation results
    vis_result = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=pose_model.cfg.data.test.type,
        show=False)

    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(vis_result, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", resized)
    cv2.waitKey(1)
