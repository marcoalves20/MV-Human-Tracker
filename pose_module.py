import cv2
import mediapipe as mp
import time
import numpy as np


class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False,
                 min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity,
                                     smooth_landmarks=smooth_landmarks, enable_segmentation=enable_segmentation,
                                     min_detection_confidence=min_detection_confidence)

    @property
    def get_landmarks(self):
        return list(mp.solutions.pose.PoseLandmark)

    def predict(self, img):
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            h, w, c = img.shape
            keypoints = []
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = lm.x * w, lm.y * h
                keypoints.append([cx, cy])
            return np.asarray(keypoints)
        else:
            return np.asarray(np.zeros((33,2)))

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
        if draw:
            cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('videos/cam01.mp4')  # make VideoCapture(0) for webcam
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        # lmList = detector.getPosition(img)
        keypoints = detector.predict(img)
        # print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
