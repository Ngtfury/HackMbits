import cv2
import mediapipe as mp
import math
import time
import numpy as np
from collections import deque
import components.brightness as bright

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
blur_toggle = False

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3]) + 1e-8
    return (A + B) / (2.0 * C)

def calculate_distance(eye_width_px):
    return (REAL_EYE_DIAMETER_MM * FOCAL_LENGTH_PX) / eye_width_px

# ------------------- Main Loop -------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    movement_flag = False
    smoothed_ear = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # ----- Blink Detection -----
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

        if prev_face_area is not None:
            rel_change = abs(face_area - prev_face_area) / (prev_face_area + 1e-8)
            if rel_change > FACE_MOVEMENT_REL_THRESH:
                movement_flag = True
        prev_face_area = face_area

        if movement_flag:
            is_eye_closed = False
            closed_start_time = None
            saw_fast_drop = False
            frame_counter = 0
        else:
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

        # ----- Distance Estimation -----
        left_corner = (int(face_landmarks.landmark[33].x * w),
                       int(face_landmarks.landmark[33].y * h))
        right_corner = (int(face_landmarks.landmark[133].x * w),
                        int(face_landmarks.landmark[133].y * h))
        eye_width_px = math.hypot(right_corner[0] - left_corner[0], right_corner[1] - left_corner[1])
        distance_cm = calculate_distance(eye_width_px) / 10

        # Toggle blur overlay if too close
        if distance_cm < 33:
            if not blur_toggle:
                blur_toggle = True
                from components.blur import BlurOverlay
                overlay = BlurOverlay(blur_radius=10)
                overlay.toggle()
        else:
            if blur_toggle:
                overlay.toggle()
            blur_toggle = False

        # ----- Draw Info -----
        for point in left_eye_points + right_eye_points:
            cv2.circle(frame, point, 1, (0, 255, 255), -1)
        cv2.putText(frame, f"Distance: {distance_cm:.1f} cm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ear_text = f"EAR: {smoothed_ear:.3f}" if smoothed_ear is not None else "EAR: -"
        thr_text = f"Thr: {threshold:.3f}" if threshold is not None else "Thr: -"
        cv2.putText(frame, f"{ear_text} {thr_text}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ----- Flash Logic -----
    elapsed_time = time.time() - start_time
    bpm = blink_count / (elapsed_time / 60.0) if elapsed_time > 0 else 0.0
    current_time = time.time()
    if elapsed_time > 5:
        if bpm < 12 and (current_time - last_flash_time > 10):
            try:
                if not blur_toggle:
                    #flash.flash_screen(color="white", duration=100, opacity=1)
                    bright.subtle_blink_pulse()
            except Exception:
                pass
            last_flash_time = current_time

    cv2.imshow("Blink & Distance Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
