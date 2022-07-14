import numpy as np
import cv2
from utils.twoview_geometry import compute_epilines, distance_point_line

INFTY_COST = 1e+5

def pose_similarity(poses_1, poses_2, calib_1, calib_2, frame=None):
    """It calculates pose similarities in second view."""

    proj1 = calib_1['K'] @ np.concatenate((calib_1['R'], calib_1['t'].reshape(3, 1)), 1)
    proj2 = calib_2['K'] @ np.concatenate((calib_2['R'], calib_2['t'].reshape(3, 1)), 1)

    similarity_poses = np.zeros((len(poses_2), len(poses_1)))

    for p1_id, p1 in enumerate(poses_1):
        pose1 = p1[:, :2]

        for p2_id, p2 in enumerate(poses_2):
            pose2 = p2[:, :2]

            # for pt in pose2:
            #     kp = (int(pt[0]), int(pt[1]))
            #     cv2.circle(frame, kp, 2, (255, 0, 0), 2)
            # resized = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
            # cv2.imshow('test', resized)
            # cv2.waitKey(0)

            points3D = np.zeros((len(pose1), 3))
            points4D = cv2.triangulatePoints(proj1, proj2, pose1.T, pose2.T)

            points3D[:, 0] = points4D[0, :] / points4D[3, :]
            points3D[:, 1] = points4D[1, :] / points4D[3, :]
            points3D[:, 2] = points4D[2, :] / points4D[3, :]

            rvec, jac = cv2.Rodrigues(calib_2['R'])
            kps_proj, jac = cv2.projectPoints(points3D, rvec, calib_2['t'], calib_2['K'], None)
            kps_proj = np.squeeze(kps_proj)

            # for pt in kps_proj:
            #     kp = (int(pt[0]), int(pt[1]))
            #     cv2.circle(frame, kp, 2, (0, 0, 255), 2)
            # # Plot image
            # resized = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
            # cv2.imshow('test', resized)
            # cv2.waitKey(0)

            cov1 = np.cov(pose2)
            cov2 = np.cov(kps_proj)
            cosine_similarity = np.sum(cov1 * cov2, axis=1) / (np.linalg.norm(cov1, axis=1) * np.linalg.norm(cov2, axis=1))

            similarity_poses[p2_id, p1_id] = np.median(cosine_similarity)

    #ss = np.argmax(similarity_poses, 1)
    return similarity_poses


def pose_cost(poses_1, poses_2, F, kps_thres=0.8):
    """It calculates pose cost by calculating pose distances from epipolar lines."""

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


def detections_cost(dets1, dets2, F, max_dist=10, n_candidates=2):
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
            # exit this loop if the distance starts to be too high or the number candidates is reached.
            if distances[idx] > max_dist or len(idxs_candidates) >= n_candidates:
                break
            else:
                idxs_candidates.append(idx)
                sel_distances.append(distances[idx])

        sel_ids.append((idxs_candidates, sel_distances))

    return cost, sel_ids


def iou(bbox, candidates):
    """
    Calculate the Intersection over Union (IoU) between the bbox and each candidate.

    Parameters
    -----------
    bbox: ndarray
        A bounding box in the format (top left x, top left y, top right x, top right y)
    candidates: ndarray
        A matrix of candidate bounding boxes (one per row) in the same format as `bbox`.
    """
    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    # determine the coordinates of the intersection rectangles
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_candidates = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections):
    """An intersection over union distance metric.

        Parameters
        ----------
        tracks : List[Tracks]
            A list of tracks.
        detections : List[bboxes]
            A list of detections.
    """
    track_indices = np.arange(len(tracks))
    detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue

        bbox = tracks[track_idx].bbox_list[-1]
        candidates = np.asarray([detections[i] for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)

    return cost_matrix