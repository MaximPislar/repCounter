import cv2
import mediapipe as mp


class PoseEstimator:
    def __init__(self,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process(self, frame_bgr):
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        return results

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        return results.pose_landmarks.landmark
