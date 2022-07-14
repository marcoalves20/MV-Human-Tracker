import cv2
from detector_module import HumanDetector
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from pathlib import Path
from utils.camera_utils import load_calibration
from tracks import Track
from scipy.optimize import linear_sum_assignment as linear_assignment
from metrics import iou_cost


def min_cost_matching(tracks, detections, max_distance):
    """Solve linear assignment problem.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    cost_matrix = iou_cost(tracks, detections)
    #cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)
    indices = np.array(indices).T

    track_indices = np.arange(len(tracks))
    detection_indices = np.arange(len(detections))

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def vis_tracks(tracks, img):
    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            bbox = track.bbox_list[-1].astype('int')
            id = track.track_id
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), track.color, 3)
            cv2.putText(img, str(id), (bbox[0] - 10, bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)


class Tracker:
    """ This is a multi-tracker.

     Attributes
    ----------
    min_iou : flaot in [0, 1]
        The minimum iou score needed in order to associate two tracks.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    tracks : List[Track]
        The list of active tracks at the current time step.
    _next_id: int
        The track id to be used in the next new track.
    """

    def __init__(self,  min_iou = 0.5, max_age=30):
        self.min_iou = min_iou
        self.max_age = max_age
        self.tracks = []
        self._next_id = 0


    def update(self, detections):
        """ It performs track management based on newly arrived detections.

         Parameters
        ----------
        detections: list[bboxes]
            Contains new detections from yolo.
        """
        if not self.tracks:
            for det in detections:
                self._initiate_track(det)
        else:
            # Run matches
            matches, unmatched_tracks, unmatched_detections = self._match(detections)

            # Update each track based on matches
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update(detections[detection_idx])
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
            for detection_idx in unmatched_detections:
                self._initiate_track(detections[detection_idx])

            self.tracks = [t for t in self.tracks if not t.is_deleted()]


    def _match(self, detections):
        matches, unmatched_tracks, unmatched_detections = min_cost_matching(self.tracks, detections,
                                                                            max_distance=1-self.min_iou)

        return matches, unmatched_tracks, unmatched_detections


    def _initiate_track(self, detection):
        self.tracks.append(Track(self._next_id, detection, self.max_age))
        self._next_id += 1


def main():
    video_dir: Path = Path("videos/wembley/")
    vid_1 = str(video_dir) + '/cam01.mp4'
    vid_2 = str(video_dir) + '/cam02.mp4'
    write_to_file = False

    yolo_det = HumanDetector(img_size=1920)
    pose_config = 'configs/hrnet_w48_coco_256x192.py'
    pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    calibration = load_calibration('calibration/')

    cap_1 = cv2.VideoCapture(vid_1)
    cap_2 = cv2.VideoCapture(vid_2)

    if write_to_file:
        size = (int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(cap_1.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter('output/tracking.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    tracker = Tracker()

    while (cap_1.isOpened()):
        ret, img1 = cap_1.read()

        if ret:
            bboxes1 = yolo_det.predict(img1)
            #scores1 = bboxes1[:, 4]

            mask = bboxes1[:, 4] > 0.5
            detections = [b[:4] for b in bboxes1[mask, :]]
            tracker.update(detections)

            vis_tracks(tracker.tracks, img1)

            scale_percent = 50  # percent of original size
            width = int(img1.shape[1] * scale_percent / 100)
            height = int(img1.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Image", resized)
            cv2.waitKey(1)

            if write_to_file:
                videoWriter.write(img1)
        else:
            break

    cap_1.release()
    if write_to_file:
        videoWriter.release()

    print(f'Tracking completed!')



if __name__ == "__main__":
    main()