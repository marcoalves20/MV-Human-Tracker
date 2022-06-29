import numpy as np
import torch
import pickle
import os
from typing import List
from utils.mv_utils import calc_pairwise_f_mats, geometry_affinity, match_als
from data_objects import Pose, Calib

def match_multiview_poses(cam_poses: List[List[Pose]], calibs: List[Calib]):
    points_set = []
    dimsGroup = [0]
    cnt = 0
    for poses in cam_poses:
        cnt += len(poses)
        dimsGroup.append(cnt)
        for p in poses:
            points_set.append(p.keypoints)

    points_set = np.array(points_set)
    pairwise_f_mats = calc_pairwise_f_mats(calibs)
    s_mat, affinity_matrix = geometry_affinity(points_set, pairwise_f_mats, dimsGroup)
    # match_mat = matchSVT(torch.from_numpy(s_mat), dimsGroup)
    match_mat, _ = match_als(s_mat, dimsGroup)
    match_mat = torch.Tensor(match_mat)

    bin_match = match_mat[:, torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).squeeze()] > 0.9
    bin_match = bin_match.reshape(s_mat.shape[0], -1)
    matched_list = [[] for i in range(bin_match.shape[1])]
    for sub_imgid, row in enumerate(bin_match):
        if row.sum() != 0:
            pid = row.numpy().argmax()
            matched_list[pid].append(sub_imgid)

    outputs = []
    for matches in matched_list:
        cam_p_idxs = []
        for idx in matches:
            cam_offset = 0
            cam_idx = 0
            for cur_cam_idx, offset in enumerate(dimsGroup):
                if offset <= idx:
                    cam_offset = offset
                    cam_idx = cur_cam_idx
                else:
                    break

            p_idx = idx - cam_offset
            cam_p_idxs.append((cam_idx, p_idx))

        if cam_p_idxs:
            outputs.append(cam_p_idxs)

    return outputs

if __name__ == '__main__':
    d1 = np.load('detections/detections_cam01.npy')
    d2 = np.load('detections/detections_cam02.npy')
    d3 = np.load('detections/detections_cam03.npy')

    # cam_list = ['cam01.pkl', 'cam02.pkl', 'cam03.pkl']
    cam_list = ['cam01.pkl', 'cam02.pkl', 'cam03.pkl']
    Cameras = []
    for camera in cam_list:
        data = pickle.load(open('calibration/'+camera, 'rb'))
        Cameras.append(Calib(data['K'], data['R'], data['t']))

    P1,P2,P3 = [], [], []
    for i in range(len(d1)):
        P1.append(Pose(pose_type=1, keypoints=np.array([np.mean([d1[i, 0], d1[i, 2]]), np.mean([d1[i, 1], d1[i, 3]])])))
    for i in range(len(d2)):
        P2.append(Pose(pose_type=1, keypoints=np.array([np.mean([d2[i, 0], d2[i, 2]]), np.mean([d2[i, 1], d2[i, 3]])])))
    for i in range(len(d3)):
        P3.append(Pose(pose_type=1, keypoints=np.array([np.mean([d3[i, 0], d3[i, 2]]), np.mean([d3[i, 1], d3[i, 3]])])))
    Poses = [P1, P2, P3]

    match_multiview_poses(Poses, Cameras)