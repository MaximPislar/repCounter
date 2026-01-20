import cv2
import numpy as np
import mediapipe as mp

from resize_func import resize_with_aspect_ratio


WINDOW_NAME = "Pose Reps"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    """
    a, b, c — точки (x, y)
    angle у вершины b
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


cap = cv2.VideoCapture(0)  # 0 — вебкамера
counter = 0
stage = None  # "up" или "down"

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB для MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Детекция
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # пример: правый локоть
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]

            angle = calculate_angle(shoulder, elbow, wrist)

            # отрисовка угла
            cv2.putText(image, str(int(angle)),
                        tuple(np.array(elbow, dtype=int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # логика подсчёта повторений
            if angle > 160:
                stage = "down"
            if angle < 40 and stage == 'down':
                stage = "up"
                counter += 1

        except:
            pass

        # Отрисовка landmarks
        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Показ
        cv2.putText(image, f'Reps: {counter}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        try:
            width, height = cv2.getWindowImageRect(WINDOW_NAME)[2:]
        except cv2.error:
            break

        # Масштабироване изображение под размер окна
        display = resize_with_aspect_ratio(image, width, height)
        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC для выхода
            break
except:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
