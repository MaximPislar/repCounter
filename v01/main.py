from video import VideoCaptureService
from pose_estimator import PoseEstimator
from rep_engine import RepEngine
from overlay import OverlayRenderer


video = VideoCaptureService(fullscreen=False)
pose = PoseEstimator()
engine = RepEngine()
overlay = OverlayRenderer(is_draw_pose=False, is_draw_info=True)

while True:
    frame = video.read()
    if frame is None:
        break

    results = pose.process(frame)
    landmarks = pose.get_landmarks(results)

    reps, angle = engine.update(landmarks, frame.shape)

    overlay.draw_pose(frame, results)
    overlay.draw_info(frame, reps, angle)

    if not video.show(frame):
        break

    if video.should_close():
        break

video.release()
