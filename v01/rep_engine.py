import numpy as np
import mediapipe as mp


class RepEngine:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.mp_pose = mp.solutions.pose

    @staticmethod
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])

        angle = abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def update(self, landmarks, frame_shape):
        if landmarks is None:
            return self.counter, None

        pose = self.mp_pose.PoseLandmark

        shoulder = landmarks[pose.RIGHT_SHOULDER.value]
        elbow = landmarks[pose.RIGHT_ELBOW.value]
        wrist = landmarks[pose.RIGHT_WRIST.value]

        if min(shoulder.visibility, elbow.visibility, wrist.visibility) < 0.6:
            return self.counter, None

        h, w, _ = frame_shape

        shoulder_xy = [shoulder.x * w, shoulder.y * h]
        elbow_xy = [elbow.x * w, elbow.y * h]
        wrist_xy = [wrist.x * w, wrist.y * h]

        angle = self.calculate_angle(shoulder_xy, elbow_xy, wrist_xy)

        if angle > 160:
            self.stage = "down"
        elif angle < 40 and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        return self.counter, angle
