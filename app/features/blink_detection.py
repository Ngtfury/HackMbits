import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from . import flash
import threading

class BlinkDetection:
    def __init__(self):
        self.running = False
        self.thread = None

        # --- Mediapipe setup ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Eye landmark indices (Mediapipe Face Mesh)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Parameters
        self.EAR_DEFAULT_THRESHOLD = 0.25
        self.SMOOTH_WINDOW = 5
        self.BASELINE_WINDOW = 150
        self.MIN_CLOSED_FRAMES = 4
        self.FACE_MOVEMENT_REL_THRESH = 0.06
        self.MIN_BLINK_DURATION = 0.03
        self.MAX_BLINK_DURATION = 0.6
        self.DERIVATIVE_DROP_REQ = -0.015

    def start(self):
        """Start blink detection in a background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            print("[BlinkDetection] Started")

    def stop(self):
        """Stop blink detection"""
        self.running = False
        print("[BlinkDetection] Stopped")

    def euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def eye_aspect_ratio(self, eye_points):
        A = self.euclidean_distance(eye_points[1], eye_points[5])
        B = self.euclidean_distance(eye_points[2], eye_points[4])
        C = self.euclidean_distance(eye_points[0], eye_points[3]) + 1e-8
        return (A + B) / (2.0 * C)

    def run(self):
        """Actual blink detection loop"""
        cap = cv2.VideoCapture(0)

        # State
        ear_queue = deque(maxlen=self.SMOOTH_WINDOW)
        baseline_queue = deque(maxlen=self.BASELINE_WINDOW)
        prev_smoothed_ear = None
        prev_face_area = None
        blink_count = 0
        frame_counter = 0
        start_time = time.time()
        last_flash_time = 0

        is_eye_closed = False
        closed_start_time = None
        saw_fast_drop = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            movement_flag = False
            smoothed_ear = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # get eye points
                left_eye_points = [(int(face_landmarks.landmark[i].x * w),
                                    int(face_landmarks.landmark[i].y * h)) for i in self.LEFT_EYE]
                right_eye_points = [(int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)) for i in self.RIGHT_EYE]

                # compute EARs
                left_ear = self.eye_aspect_ratio(left_eye_points)
                right_ear = self.eye_aspect_ratio(right_eye_points)
                raw_ear = (left_ear + right_ear) / 2.0

                # bounding box area
                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                face_area = max(1, (max_x - min_x) * (max_y - min_y))

                if prev_face_area is not None:
                    rel_change = abs(face_area - prev_face_area) / (prev_face_area + 1e-8)
                    if rel_change > self.FACE_MOVEMENT_REL_THRESH:
                        movement_flag = True
                prev_face_area = face_area

                if movement_flag:
                    is_eye_closed = False
                    closed_start_time = None
                    saw_fast_drop = False
                    frame_counter = 0
                else:
                    ear_queue.append(raw_ear)
                    if len(ear_queue) >= 3:
                        smoothed_ear = float(np.median(np.array(ear_queue)))
                    else:
                        smoothed_ear = float(np.mean(np.array(ear_queue)))

                    if smoothed_ear is not None and smoothed_ear > 0.18:
                        baseline_queue.append(smoothed_ear)

                    if len(baseline_queue) >= 30:
                        baseline = float(np.mean(baseline_queue))
                        threshold = max(self.EAR_DEFAULT_THRESHOLD * 0.6, baseline * 0.70)
                    else:
                        threshold = self.EAR_DEFAULT_THRESHOLD

                    derivative = smoothed_ear - prev_smoothed_ear if prev_smoothed_ear is not None else 0.0

                    if smoothed_ear < threshold:
                        frame_counter += 1
                        if derivative < self.DERIVATIVE_DROP_REQ:
                            saw_fast_drop = True
                        if not is_eye_closed:
                            is_eye_closed = True
                            closed_start_time = time.time()
                    else:
                        if is_eye_closed:
                            closed_duration = time.time() - (closed_start_time or time.time())
                            if (frame_counter >= self.MIN_CLOSED_FRAMES and
                                saw_fast_drop and
                                self.MIN_BLINK_DURATION <= closed_duration <= self.MAX_BLINK_DURATION):
                                blink_count += 1
                            is_eye_closed = False
                            closed_start_time = None
                            saw_fast_drop = False
                            frame_counter = 0

                    prev_smoothed_ear = smoothed_ear

                # Draw debug
                for point in left_eye_points + right_eye_points:
                    cv2.circle(frame, point, 1, (0, 255, 255), -1)

                cv2.putText(frame, f"EAR: {smoothed_ear:.3f}" if smoothed_ear else "EAR: -", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elapsed_time = time.time() - start_time
            bpm = blink_count / (elapsed_time / 60.0) if elapsed_time > 0 else 0.0

            current_time = time.time()
            if elapsed_time > 5:
                if bpm < 12 and (current_time - last_flash_time > 10):
                    try:
                        flash.flash_screen(color="white", duration=100, opacity=1)
                    except Exception:
                        pass
                    last_flash_time = current_time

            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("[BlinkDetection] Loop ended")
