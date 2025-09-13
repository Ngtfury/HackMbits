import cv2
import mediapipe as mp
import math
import time
import numpy as np
from collections import deque
import components.brightness as bright
import threading
import keyboard
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont, QPen
import sys

# ------------------- MediaPipe Setup -------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------- Blink Detection Setup -------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_DEFAULT_THRESHOLD = 0.25
SMOOTH_WINDOW = 5
BASELINE_WINDOW = 150
MIN_CLOSED_FRAMES = 4
FACE_MOVEMENT_REL_THRESH = 0.06
MIN_BLINK_DURATION = 0.03
MAX_BLINK_DURATION = 0.6
DERIVATIVE_DROP_REQ = -0.015

ear_queue = deque(maxlen=SMOOTH_WINDOW)
baseline_queue = deque(maxlen=BASELINE_WINDOW)
prev_smoothed_ear = None
prev_face_area = None

blink_count = 0
frame_counter = 0
is_eye_closed = False
closed_start_time = None
saw_fast_drop = False
start_time = time.time()
last_flash_time = 0

# ------------------- Distance Estimation Setup -------------------
REAL_EYE_DIAMETER_MM = 24.0
FOCAL_LENGTH_PX = 600
blur_toggle = False  # <-- stateful toggle

# BPM history (for graph)
bpm_history = deque(maxlen=60)

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3]) + 1e-8
    return (A + B) / (2.0 * C)

def calculate_distance(eye_width_px):
    return (REAL_EYE_DIAMETER_MM * FOCAL_LENGTH_PX) / eye_width_px

# ------------------- PySide HUD Overlay -------------------
class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().geometry())

        self.current_bpm = 0
        self.current_distance = 0
        self.current_blinks = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)  # refresh 10 FPS
        self.hide()

    def update_data(self, bpm, distance, blinks):
        self.current_bpm = bpm
        self.current_distance = distance
        self.current_blinks = blinks

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 160))  # semi-transparent background

        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 0))

        painter.drawText(50, 100, f"Blink Count: {self.current_blinks}")
        painter.drawText(50, 150, f"BPM: {self.current_bpm:.1f}")
        painter.drawText(50, 200, f"Distance: {self.current_distance:.1f} cm")

        # Draw BPM graph
        if len(bpm_history) > 1:
            pen = QPen(QColor(0, 255, 255))
            pen.setWidth(3)
            painter.setPen(pen)
            graph_width = 400
            graph_height = 100
            x_start, y_start = 50, 250

            max_bpm = max(max(bpm_history), 1)
            points = []
            for i, bpm in enumerate(bpm_history):
                x = x_start + (i / len(bpm_history)) * graph_width
                y = y_start + graph_height - (bpm / max_bpm) * graph_height
                points.append((x, y))

            for i in range(1, len(points)):
                painter.drawLine(points[i-1][0], points[i-1][1], points[i][0], points[i][1])

# ------------------- Background Capture Thread -------------------
def capture_loop(overlay: OverlayWindow):
    global prev_smoothed_ear, prev_face_area, blink_count, frame_counter, is_eye_closed
    global closed_start_time, saw_fast_drop, last_flash_time, blur_toggle

    cap = cv2.VideoCapture(0)
    overlay_instance = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        smoothed_ear = None
        distance_cm = 0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye_points = [(int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye_points = [(int(face_landmarks.landmark[i].x * w),
                                 int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            raw_ear = (left_ear + right_ear) / 2.0

            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            face_area = max(1, (max_x - min_x) * (max_y - min_y))

            movement_flag = False
            if prev_face_area is not None:
                rel_change = abs(face_area - prev_face_area) / (prev_face_area + 1e-8)
                if rel_change > FACE_MOVEMENT_REL_THRESH:
                    movement_flag = True
            prev_face_area = face_area

            if not movement_flag:
                ear_queue.append(raw_ear)
                smoothed_ear = float(np.median(np.array(ear_queue))) if len(ear_queue) >= 3 else float(np.mean(np.array(ear_queue)))

                if smoothed_ear is not None and smoothed_ear > 0.18:
                    baseline_queue.append(smoothed_ear)

                baseline = float(np.mean(baseline_queue)) if len(baseline_queue) >= 30 else EAR_DEFAULT_THRESHOLD
                threshold = max(EAR_DEFAULT_THRESHOLD * 0.6, baseline * 0.70)

                derivative = 0 if prev_smoothed_ear is None else smoothed_ear - prev_smoothed_ear

                if smoothed_ear < threshold:
                    frame_counter += 1
                    if derivative < DERIVATIVE_DROP_REQ:
                        saw_fast_drop = True
                    if not is_eye_closed:
                        is_eye_closed = True
                        closed_start_time = time.time()
                else:
                    if is_eye_closed:
                        closed_duration = time.time() - (closed_start_time or time.time())
                        if frame_counter >= MIN_CLOSED_FRAMES and saw_fast_drop and MIN_BLINK_DURATION <= closed_duration <= MAX_BLINK_DURATION:
                            blink_count += 1
                        is_eye_closed = False
                        closed_start_time = None
                        saw_fast_drop = False
                        frame_counter = 0

                prev_smoothed_ear = smoothed_ear

            # Distance estimation
            left_corner = (int(face_landmarks.landmark[33].x * w),
                           int(face_landmarks.landmark[33].y * h))
            right_corner = (int(face_landmarks.landmark[133].x * w),
                            int(face_landmarks.landmark[133].y * h))
            eye_width_px = math.hypot(right_corner[0] - left_corner[0], right_corner[1] - left_corner[1])
            distance_cm = calculate_distance(eye_width_px) / 10

            # --- Blur Toggle Logic ---
            if distance_cm < 33:
                if not blur_toggle:
                    blur_toggle = True
                    from components.blur import BlurOverlay
                    overlay_instance = BlurOverlay(blur_radius=10)
                    overlay_instance.toggle()
            else:
                if blur_toggle and overlay_instance:
                    overlay_instance.toggle()
                    overlay_instance = None
                blur_toggle = False

        elapsed_time = time.time() - start_time
        bpm = blink_count / (elapsed_time / 60.0) if elapsed_time > 0 else 0.0
        bpm_history.append(bpm)

        if elapsed_time > 5:
            if bpm < 12 and (time.time() - last_flash_time > 10):
                try:
                    if not blur_toggle:
                        bright.subtle_blink_pulse()
                except Exception:
                    pass
                last_flash_time = time.time()

        overlay.update_data(bpm, distance_cm, blink_count)

    cap.release()

def hotkey_loop(overlay: OverlayWindow):
    while True:
        keyboard.wait("ctrl+shift+b")
        if overlay.isVisible():
            overlay.hide()
        else:
            overlay.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    threading.Thread(target=capture_loop, args=(overlay,), daemon=True).start()
    threading.Thread(target=hotkey_loop, args=(overlay,), daemon=True).start()
    sys.exit(app.exec())
 