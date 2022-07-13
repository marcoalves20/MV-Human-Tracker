import numpy as np
import cv2
from utils.camera_utils import load_calibration
from utils.twoview_geometry import fundamental_from_poses
from metrics import pose_similarity, pose_cost, detections_cost
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
    sel_indexes = {}
    dist_array = {}
    all_cost_poses = []
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
                cost, sel_ids = detections_cost(positions_undist1, positions_undist2, F,
                                                                        view1, view2, max_dist, n_candidates)
                sel_indexes[view1][view2] = sel_ids
                #row_ind, col_ind = linear_sum_assignment(cost)
                dist_array[view1][view2] = cost

                # Calculate pose cost
                poses_2 = poses[view2]
                cost_poses = pose_cost(poses_1, poses_2, F, kps_thres=0.8)
                all_cost_poses.append(cost_poses)

    return sel_indexes, all_cost_poses[0]


def filter_matching(matches, cost_poses):

    views = [k for k in matches.keys()]
    indexes = []
    tmp_list = list(matches[views[0]].values())
    for i, id in enumerate(tmp_list[0]):
        try:
            if id[0] != []:
                indexes.append([i, id[0]])
        except:
            pass

    indexes = np.array(indexes, dtype=object)

    # Remove duplicate values by checking the cost_poses
    sel_ids = np.array([i[0] for i in indexes[:, 1]])
    dump_val = 10000
    while len(sel_ids) != len(set(sel_ids)): # Repeat as long as sel_ids contains duplicate values
        for i in range(len(indexes)):
            dup_ids = np.where(sel_ids == sel_ids[i])[0]
            if len(dup_ids) > 1:
                ids_view1 = indexes[dup_ids, 0]
                costs = cost_poses[list(ids_view1), sel_ids[i]]
                sorted_ids = np.argsort(costs)

                for j in range(1, len(sorted_ids)):
                    tmp_id = dup_ids[sorted_ids[j]]
                    indexes[tmp_id, :][1] = indexes[tmp_id, :][1][1:] # remove first element
                    if len(indexes[tmp_id, :][1]) != 0:
                        sel_ids[tmp_id] = indexes[tmp_id, :][1][0]
                    else:
                        sel_ids[tmp_id] = dump_val
                        dump_val += 1

    indexes[:, 1] = sel_ids
    lst = list(np.where(indexes >= 10000))
    indexes = np.delete(indexes, lst[0], axis=0)

    # Remove duplicate values
    # for i in range(indexes.shape[0]):
    #     id1 = indexes[i, 1]
    #     for j in range(indexes.shape[0]):
    #         # Case we have duplicate values
    #         if i!=j and indexes[j, 1] == id1:
    #
    #             if matches['cam01']['cam02'][indexes[i, 0]][1][0] < matches['cam01']['cam02'][indexes[j, 0]][1][0] and len(matches['cam01']['cam02'][indexes[j, 0]][0]) > 1:
    #                 matches['cam01']['cam02'][indexes[j, 0]][1].pop(0)
    #                 matches['cam01']['cam02'][indexes[j, 0]][0].pop(0)
    #                 indexes[j, 1] = matches['cam01']['cam02'][indexes[j, 0]][0][0]
    #             elif matches['cam01']['cam02'][indexes[i, 0]][1][0] > matches['cam01']['cam02'][indexes[j, 0]][1][0] and len(matches['cam01']['cam02'][indexes[i, 0]][0]) > 1:
    #                 matches['cam01']['cam02'][indexes[i, 0]][1].pop(0)
    #                 matches['cam01']['cam02'][indexes[i, 0]][0].pop(0)
    #                 indexes[i, 1] = matches['cam01']['cam02'][indexes[i, 0]][0][0]

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
    #img3 = np.load('detections/cam03_1stframe.npy')
    with open("detections/detections_dict_cam01.pkl", "rb") as f:
        dict1 = pickle.load(f)
    with open("detections/detections_dict_cam02.pkl", "rb") as f:
        dict2 = pickle.load(f)
    # with open("detections/detections_dict_cam03.pkl", "rb") as f:
    #     dict3 = pickle.load(f)
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

    similarity_poses = pose_similarity(p1, p2, calibration['cam01'], calibration['cam02'], img2.copy())

    matches, cost_poses = find_candidate_matches(detections, poses, views, calibration, max_dist=15, n_candidates=4, verbose=0)
    sorted_idx = filter_matching(matches, cost_poses)

    new_d1 = d1[list(sorted_idx[:, 0]), :]
    new_d2 = d2[list(sorted_idx[:, 1]), :]
    preview(img1, img2, new_d1, new_d2)

    print(1)



