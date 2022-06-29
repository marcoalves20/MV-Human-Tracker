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



def find_candidate_matches(detections, views, calibration, max_dist=10, n_candidates=2,
                           verbose=0):
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

    for view, ds in detections.items():
        K = np.array(calibration[view]['K'])
        dist = np.array(calibration[view]['dist'])
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

                positions_undist2 = np.reshape([detection.position
                                                for detection in detections[view2]], (-1, 2))

                _, lines2 = compute_epilines(positions_undist1, None, F)
                cost = np.zeros((len(positions_undist1), len(positions_undist2)))

                for i1, line in enumerate(lines2):

                    distances = [distance_point_line(x, line) for x in positions_undist2]
                    cost[i1,:] = distances
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

                    sel_indexes[view1][view2].append((idxs_candidates, sel_distances))
                row_ind, col_ind = linear_sum_assignment(cost)
                dist_array[view1][view2] = cost
    return sel_indexes

def filter_matching(views, matches):
    indexes = []
    for i, id in enumerate(matches['cam01']['cam02']):
        try:
            indexes.append([i,id[0][0]])
        except:
            pass
    return np.array(indexes)



def preview(img1,img2,d1,d2):
    for i in range(len(d1)):
        img1 = cv2.rectangle(img1, (d1[i, 0], d1[i, 1]), (d1[i, 2], d1[i, 3]), (0, 255, 0), 3)
        img1 = cv2.putText(img=img1,text=str(i),org=(d1[i, 0], d1[i, 1]),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=3.0,color=(125, 246, 55),thickness=3)
    for i in range(len(d2)):
        img2=cv2.rectangle(img2, (d2[i, 0], d2[i, 1]), (d2[i, 2], d2[i, 3]), (0, 255, 0), 3)
        img2=cv2.putText(img=img2,text=str(i),org=(d2[i, 0], d2[i, 1]),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=3.0,color=(125, 246, 55),thickness=3)
    import matplotlib.pyplot as plt
    plt.imshow(np.hstack([img1,img2]))
    plt.show()

if __name__ == '__main__':
    d1= np.load('detections/detections_cam01.npy')
    d2 = np.load('detections/detections_cam02.npy')
    d3 = np.load('detections/detections_cam03.npy')
    img1 = np.load('detections/cam01_1stframe.npy')
    img2 = np.load('detections/cam02_1stframe.npy')
    preview(img1,img2,d1,d2)

    views = ['cam01', 'cam02']
    D1,D2, D3 = [], [], []
    for i in range(len(d1)):
        D1.append(Detection2D(position=np.array([np.mean([d1[i,0],d1[i,2]]),np.mean([d1[i,1],d1[i,3]])]), confidence=d1[i,4],view=views[0]))
    for i in range(len(d2)):
        D2.append(Detection2D(position=np.array([np.mean([d2[i,0],d2[i,2]]),np.mean([d2[i,1],d2[i,3]])]), confidence=d2[i,4],view=views[1]))
    # for i in range(len(d3)):
    #     D3.append(Detection2D(position=np.array([np.mean([d3[i,0],d3[i,2]]),np.mean([d3[i,1],d3[i,3]])]), confidence=d3[i,4],view=views[2]))
    calibration = load_calibration('calibration/')
    detections = {'cam01': D1, 'cam02':D2}

    matches = find_candidate_matches(detections, views, calibration, max_dist=10, n_candidates=2,
                           verbose=0)
    sorted_idx = filter_matching(views, matches)

    new_d1 = d1[sorted_idx[:,0],:]
    new_d2 = d2[sorted_idx[:,1],:]
    preview(img1, img2, new_d1, new_d2)

    print(1)



