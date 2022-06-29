from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple
import copy
import numpy as np

class KpsFormat(Enum):
    Mediapipe = 1

@dataclass
class Pose:
    pose_type: KpsFormat
    keypoints: np.ndarray
    # keypoints_score: Optional[np.ndarray]
    # box: Optional[np.ndarray]


@dataclass
class Calib:
    K: np.ndarray  # 3x3
    R: np.ndarray  # 3x3
    t: np.ndarray  # 1x2

    @property
    def cam_loc(self):
        return -self.Rt[:3, :3].T @ self.Rt[:3, 3]

    @property
    def P(self):
        return self.K @ np.hstack((self.R, self.t.reshape(3, 1)))

    @property
    def Rt(self):
        return np.hstack((self.R, self.t[...,np.newaxis]))

