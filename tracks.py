import numpy as np
import random


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

# TODO: Add tentative state. Currently only Confirmed and Deleted trackstate is used.

class Track:
    """ A single target track.

     Attributes
    ----------
    track_id : int
        A unique track identifier.
    bbox_list : list[bboxes]
        A list containing the position of the track in the form of bboxes (in tlbr form).
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    state : TrackState
        The current track state.
    """

    def __init__(self, track_id, bbox, max_age):
        self.track_id = track_id
        self.bbox_list = [bbox]
        self.age = 1
        self.time_since_update = 0
        self._max_age = max_age
        self.state = TrackState.Confirmed

        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        self.color = [b, g, r]


    def to_mean_pos(self):
        """ Get current position as the middle of bbox."""
        ret = np.array([np.mean([self.bbox_list[-1][0], self.bbox_list[-1][2]]),
                        np.mean([self.bbox_list[-1][1], self.bbox_list[-1][3]])])
        return ret


    def to_tlwh(self):
        """ Get current position in bounding box format `(top left x, top left y, width, height)`. """
        w = self.bbox_list[-1][2] - self.bbox_list[-1][0]
        h = self.bbox_list[-1][3] - self.bbox_list[-1][1]
        ret = np.array([self.bbox_list[-1][0], self.bbox_list[-1][1], w, h])
        return ret


    def update(self, bbox):
        self.bbox_list.append(bbox)
        self.age += 1
        self.time_since_update = 0


    def mark_missed(self):
        """ Mark this track as missed (no association at the current time step)."""
        self.time_since_update += 1
        if self.time_since_update > self._max_age:
            self.state = TrackState.Deleted


    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed


    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
