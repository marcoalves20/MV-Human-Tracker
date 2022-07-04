import os
import sys
import time
import numpy as np
import itertools
import cv2
import time
from utils.camera_utils import load_calibration
from utils.twoview_geometry import fundamental_from_poses, compute_epilines, distance_point_line
from scipy.optimize import linear_sum_assignment
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Detection(object):

    def __init__(self, index=None, position=None, confidence=0.5, datetime=None, id=None):
        self.index = index
        self.position = position
        self.confidence = confidence
        self.datetime = datetime
        self.id = id

    def __str__(self):
        return """{self.__class__.__name__}(index={self.index}, confidence={self.confidence}, datetime={self.datetime}, position={self.position})""".format(
            self=self)


class Detection2D(Detection):

    def __init__(self, view=None, index=None, position=None,
                 position_undist=None, confidence=0.5,
                 datetime=None, id=None):
        super(Detection2D, self).__init__(index, position, confidence, datetime, id)
        self.view = view
        self.position_undist = position_undist
        self.node = None

    def __str__(self):
        return """{self.__class__.__name__}(view={self.view}, index={self.index}, confidence={self.confidence}, datetime={self.datetime}, position={self.position}, position_undist={self.position_undist})""".format(
            self=self)


def calc_cost_poses(poses_1, poses_2, F, kps_thres=0.8):

    cost_poses = np.zeros((len(poses_1), len(poses_2)))

    for i, pose1 in enumerate(poses_1):
        if np.all(pose1 == 0):
            cost_poses[i, :] = np.nan
            continue

        _, lines = compute_epilines(pose1[:, :2], None, F)  # 33 lines (equal to the number of keypoints)

        # calculate distance from epipolar lines for each pose in second view
        for id, pose2 in enumerate(poses_2):
            distances = []
            for pt_id in range(pose2.shape[0]):
                if pose2[pt_id, 2] > kps_thres and pose1[pt_id, 2] > kps_thres:
                    distances.append(distance_point_line(pose2[pt_id, :2], lines[pt_id, :]))

            if len(distances) != 0:
                cost_poses[i, id] = np.nanmedian(distances)
            else:
                cost_poses[i, id] = np.nan

    return cost_poses


def calc_cost_detections(dets1, dets2, F, view1, view2, max_dist=10, n_candidates=2, verbose=0):
    """ It calculates the cost matrix based on bbox detections and returns candidate matches based on this."""

    cost = np.zeros((len(dets1), len(dets2)))
    sel_ids = []

    _, lines = compute_epilines(dets1, None, F)

    for i1, line in enumerate(lines):

        distances = [distance_point_line(x, line) for x in dets2]
        cost[i1, :] = distances
        idx_sorted = np.argsort(distances)
        idxs_candidates = []
        sel_distances = []
        for idx in idx_sorted:
            # exit this loop if the distance start to be
            # o high or the number candidates is reached
            if verbose == 2:
                print("{}-{} {}-{} {:0.2f}".format(view1, i1, view2, idx, distances[idx]))
            if distances[idx] > max_dist:
                if verbose == 2:
                    print("{}-{} {}-{} discarded because of distance {:0.2f}".format(view1, i1, view2, idx,
                                                                                     distances[idx]))
                else:
                    break
            elif len(idxs_candidates) >= n_candidates:
                if verbose == 2:
                    print("{}-{} {}-{} discarded because of number of candidates reached.".format(view1, i1,
                                                                                                  view2,
                                                                                                  idx))
                else:
                    break
            else:
                if verbose == 2:
                    print("{}-{} {}-{} selected distance {:0.2f}".format(view1, i1, view2, idx,
                                                                         distances[idx]))
                idxs_candidates.append(idx)
                sel_distances.append(distances[idx])

        sel_ids.append((idxs_candidates, sel_distances))

    return cost, sel_ids


def find_candidate_matches(detections, poses, views, calibration, max_dist=10, n_candidates=2, verbose=0):
    """
    Given a detection in one view, find the best candidates detections on the other views

    Parameters
    ----------
    detections : dict of lists of objects of type Detection2D
        {'view1':[Detection1, Detection2, ...], 'view2':[...]}
    views : list
        list cotaining the name of the views i.e. ['view1', 'view2', ...]
    calibration : dict
        extrinsic and instrinsic parameters {'view1':{'R':.., 't':.., 'K':..., 'dist':...}}
    max_dist : float
        a detection is considered a candidate if its distance to a epiline is less than this
    n_candidates : int
        max number of candidates per detection in each view
    """

    # for view, ds in detections.items():
    #     K = np.array(calibration[view]['K'])
    #     dist = np.array(calibration[view]['dist'])
        # for d in ds:
        #     d.position_undist = cv2.undistortPoints(np.reshape(d.position, (1, 2)), K, dist, P=K)[0].squeeze()

    sel_indexes = {}
    dist_array = {}
    for view1 in views:

        sel_indexes[view1] = {}
        dist_array[view1] = {}

        K1 = np.array(calibration[view1]['K'])
        R1 = np.array(calibration[view1]['R'])
        t1 = np.array(calibration[view1]['t'])

        if len(detections[view1]) == 0:
            continue

        positions_undist1 = np.reshape([detection.position
                                        for detection in detections[view1]], (-1, 2))

        poses_1 = poses[view1]

        for view2 in views:

            if view1 != view2:

                sel_indexes[view1][view2] = []

                K2 = np.array(calibration[view2]['K'])
                R2 = np.array(calibration[view2]['R'])
                t2 = np.array(calibration[view2]['t'])

                F = fundamental_from_poses(K1, R1, t1, K2, R2, t2)

                if len(detections[view2]) == 0:
                    sel_indexes[view1][view2] = [([], [])] * len(detections[view1])
                    continue

                # Calculate detection cost
                positions_undist2 = np.reshape([detection.position
                                                for detection in detections[view2]], (-1, 2))
                cost, sel_ids = calc_cost_detections(positions_undist1, positions_undist2, F,
                                                                        view1, view2, max_dist, n_candidates, verbose)
                sel_indexes[view1][view2] = sel_ids
                row_ind, col_ind = linear_sum_assignment(cost)
                dist_array[view1][view2] = cost

                # Calculate pose cost
                poses_2 = poses[view2]
                cost_poses = calc_cost_poses(poses_1, poses_2, F, kps_thres=0.8)

    return sel_indexes


def filter_matching(views, matches):
    indexes = []
    for i, id in enumerate(matches['cam01']['cam02']):
        try:
            indexes.append([i,id[0][0]])
        except:
            pass

    indexes = np.array(indexes)

    # Remove duplicate values
    for i in range(indexes.shape[0]):
        id1 = indexes[i, 1]
        for j in range(indexes.shape[0]):
            # Case we have duplicate values
            if i!=j and indexes[j, 1] == id1:

                if matches['cam01']['cam02'][indexes[i, 0]][1][0] < matches['cam01']['cam02'][indexes[j, 0]][1][0] and len(matches['cam01']['cam02'][indexes[j, 0]][0]) > 1:
                    matches['cam01']['cam02'][indexes[j, 0]][1].pop(0)
                    matches['cam01']['cam02'][indexes[j, 0]][0].pop(0)
                    indexes[j, 1] = matches['cam01']['cam02'][indexes[j, 0]][0][0]
                elif matches['cam01']['cam02'][indexes[i, 0]][1][0] > matches['cam01']['cam02'][indexes[j, 0]][1][0] and len(matches['cam01']['cam02'][indexes[i, 0]][0]) > 1:
                    matches['cam01']['cam02'][indexes[i, 0]][1].pop(0)
                    matches['cam01']['cam02'][indexes[i, 0]][0].pop(0)
                    indexes[i, 1] = matches['cam01']['cam02'][indexes[i, 0]][0][0]


    return indexes


def preview(img1, img2, d1, d2):

    d1 = d1.astype(int)
    d2 = d2.astype(int)
    for i in range(len(d1)):
        img1 = cv2.rectangle(img1, (d1[i, 0], d1[i, 1]), (d1[i, 2], d1[i, 3]), (0, 255, 0), 3)
        img1 = cv2.putText(img=img1,text=str(i),org=(d1[i, 0], d1[i, 1]),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=3.0,color=(125, 246, 55),thickness=3)
    for i in range(len(d2)):
        img2=cv2.rectangle(img2, (d2[i, 0], d2[i, 1]), (d2[i, 2], d2[i, 3]), (0, 255, 0), 3)
        img2=cv2.putText(img=img2,text=str(i),org=(d2[i, 0], d2[i, 1]),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=3.0,color=(125, 246, 55),thickness=3)

    t = cv2.hconcat([img1, img2])
    resized = cv2.resize(t, (int(2 * img1.shape[1] / 4), int(img1.shape[0] / 4)), interpolation=cv2.INTER_AREA)
    cv2.imshow('test', resized)
    key = cv2.waitKey(0)
    if key:
        cv2.destroyAllWindows()
    # plt.imshow(np.hstack([img1,img2]))
    # plt.show()


def previewDetDict(img1, img2, dict1, dict2, kps_thres=0.7):
    poses = dict1['poses']
    dets = dict1['detections'].astype('int')
    for i in range(len(dets)):
        # Draw bboxes
        cv2.rectangle(img1, (dets[i, 0], dets[i, 1]), (dets[i, 2], dets[i, 3]), (0, 255, 0), 3)
        cv2.putText(img=img1,text=str(i),org=(dets[i, 0], dets[i, 1]),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=3.0,
                    color=(125, 246, 55),thickness=3)

        # Draw poses
        for pose in poses:
            for kpt in pose:
                if kpt[2] > kps_thres:
                    cv2.circle(img1, (int(kpt[0]), int(kpt[1])), 3, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(img1, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), cv2.FILLED)

    poses = dict2['poses']
    dets = dict2['detections'].astype('int')
    for i in range(len(dets)):
        # Draw bboxes
        cv2.rectangle(img2, (dets[i, 0], dets[i, 1]), (dets[i, 2], dets[i, 3]), (0, 255, 0), 3)
        cv2.putText(img=img2, text=str(i), org=(int(dets[i, 0]), int(dets[i, 1])), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                           fontScale=3.0, color=(125, 246, 55), thickness=3)

        # Draw poses
        for pose in poses:
            for kpt in pose:
                if kpt[2] > kps_thres:
                    cv2.circle(img2, (int(kpt[0]), int(kpt[1])), 3, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(img2, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), cv2.FILLED)

    t = cv2.hconcat([img1, img2])
    resized = cv2.resize(t, (int(2 * img1.shape[1]/4), int(img1.shape[0]/4 )), interpolation=cv2.INTER_AREA)
    cv2.imshow('test', resized)
    key = cv2.waitKey(0)
    if key:
        cv2.destroyAllWindows()
    # plt.imshow(np.hstack([img1, img2]))
    # plt.show()


if __name__ == '__main__':

    img1 = np.load('detections/cam01_1stframe.npy')
    img2 = np.load('detections/cam02_1stframe.npy')
    with open("detections/detections_dict_cam01.pkl", "rb") as f:
        dict1 = pickle.load(f)
    with open("detections/detections_dict_cam02.pkl", "rb") as f:
        dict2 = pickle.load(f)
    previewDetDict(img1.copy(), img2.copy(), dict1, dict2)

    views = ['cam01', 'cam02']
    calibration = load_calibration('calibration/')

    # Use the centre of each bbox as detection.
    d1 = dict1['detections']
    d2 = dict2['detections']
    D1, D2 = [], []
    for i in range(len(d1)):
        D1.append(Detection2D(position=np.array([np.mean([d1[i, 0], d1[i, 2]]), np.mean([d1[i, 1], d1[i, 3]])]),
                              confidence=d1[i, 4], view=views[0]))
    for i in range(len(d2)):
        D2.append(Detection2D(position=np.array([np.mean([d2[i, 0], d2[i, 2]]), np.mean([d2[i, 1], d2[i, 3]])]),
                              confidence=d2[i, 4], view=views[1]))
    detections = {'cam01': D1, 'cam02':D2}

    p1 = dict1['poses']
    p2 = dict2['poses']
    poses = {'cam01': p1, 'cam02': p2}

    matches = find_candidate_matches(detections, poses, views, calibration, max_dist=10, n_candidates=2, verbose=0)
    sorted_idx = filter_matching(views, matches)

    new_d1 = d1[sorted_idx[:, 0], :]
    new_d2 = d2[sorted_idx[:, 1], :]
    preview(img1, img2, new_d1, new_d2)

    print(1)



