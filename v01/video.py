import cv2
import numpy as np


class VideoCaptureService:
    def __init__(self, camera_id=0, window_name="Pose Reps", fullscreen=True):
        self.cap = cv2.VideoCapture(camera_id)
        self.window_name = window_name

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

    def resize_with_aspect_ratio(self, image, target_w, target_h):
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # создаём чёрный фон под размер окна
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # центрируем изображение
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def show(self, frame):
        try:
            w, h = cv2.getWindowImageRect(self.window_name)[2:]
            display = self.resize_with_aspect_ratio(frame, w, h)
            cv2.imshow(self.window_name, display)
        except cv2.error:
            return False
        return True

    def should_close(self):
        return cv2.waitKey(1) & 0xFF == 27

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
