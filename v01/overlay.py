import cv2
import mediapipe as mp


class OverlayRenderer:
    def __init__(self, is_draw_pose=True, is_draw_info=True):
        self.is_draw_pose = is_draw_pose
        self.is_draw_info = is_draw_info
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_pose(self, frame, results):
        if not self.is_draw_pose:
            return

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

    def draw_info(self, frame, reps, angle):
        if not self.is_draw_info:
            return

        cv2.putText(frame, f"Reps: {reps}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if angle is not None:
            cv2.putText(frame, f"Angle: {int(angle)}", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
