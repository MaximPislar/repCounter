import cv2
import numpy as np


def resize_with_aspect_ratio(image, target_w, target_h):
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


def main():
    window_name = "Camera"
    window_w, window_h = 1000, 700

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_w, window_h)

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                width, height = cv2.getWindowImageRect(window_name)[2:]
            except cv2.error:
                break

            display = resize_with_aspect_ratio(frame, width, height)
            cv2.imshow(window_name, display)

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("работа завершена")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
