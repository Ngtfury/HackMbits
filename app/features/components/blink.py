import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
import flash

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye landmark indices (Mediapipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye outline
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye outline

def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(eye_points):
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3]) + 1e-8
    ear = (A + B) / (2.0 * C)
    return ear

# --- PARAMETERS you can tune ---
EAR_DEFAULT_THRESHOLD = 0.25   # fxallback if dynamic baseline not ready
SMOOTH_WINDOW = 5              # median/mean smoothing window (frames)
BASELINE_WINDOW = 150          # frames for estimating "eyes-open" baseline
MIN_CLOSED_FRAMES = 4          # consecutive frames below threshold required
FACE_MOVEMENT_REL_THRESH = 0.06 # relative change in face area to consider "movement"
MIN_BLINK_DURATION = 0.03     # seconds (ignore extremely tiny dips)
MAX_BLINK_DURATION = 0.6      # seconds (ignore too-long closures)
DERIVATIVE_DROP_REQ = -0.015  # require a reasonably fast downward slope to start blink
# --------------------------------

ear_queue = deque(maxlen=SMOOTH_WINDOW)
baseline_queue = deque(maxlen=BASELINE_WINDOW)
prev_smoothed_ear = None
prev_face_area = None

cap = cv2.VideoCapture(0)
blink_count = 0
frame_counter = 0
start_time = time.time()
last_flash_time = 0

# state for blink detection
is_eye_closed = False
closed_start_time = None
saw_fast_drop = False

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

        # get eye points (pixel coords)
        left_eye_points = [(int(face_landmarks.landmark[i].x * w),
                            int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye_points = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

        # compute EARs
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        raw_ear = (left_ear + right_ear) / 2.0

        # --- face bounding box & movement check ---
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

        # if movement detected, skip updating EAR smoothing/baseline for this frame
        if movement_flag:
            # reset transient blink detection (to avoid false counting during motion)
            is_eye_closed = False
            closed_start_time = None
            saw_fast_drop = False
            frame_counter = 0
        else:
            # update smoothing window
            ear_queue.append(raw_ear)
            # use median smoothing to reduce spikes, fallback to mean if queue small
            if len(ear_queue) >= 3:
                smoothed_ear = float(np.median(np.array(ear_queue)))
            else:
                smoothed_ear = float(np.mean(np.array(ear_queue)))

            # update baseline only when eyes appear open (helps avoid blinks in baseline)
            if smoothed_ear is not None and smoothed_ear > 0.18:
                baseline_queue.append(smoothed_ear)

            # compute dynamic threshold: use baseline if available
            if len(baseline_queue) >= 30:
                baseline = float(np.mean(baseline_queue))
                threshold = max(EAR_DEFAULT_THRESHOLD * 0.6, baseline * 0.70)
            else:
                threshold = EAR_DEFAULT_THRESHOLD

            # derivative (smoothed)
            if prev_smoothed_ear is None:
                derivative = 0.0
            else:
                derivative = smoothed_ear - prev_smoothed_ear

            # Blink logic:
            # require BOTH a reasonably fast drop (derivative negative enough) AND
            # consecutive frames below threshold for a short time.
            if smoothed_ear < threshold:
                frame_counter += 1
                # fast drop detection (only set once per closure)
                if derivative < DERIVATIVE_DROP_REQ:
                    saw_fast_drop = True

                if not is_eye_closed:
                    # mark start of closure
                    is_eye_closed = True
                    closed_start_time = time.time()
            else:
                
                if is_eye_closed:
                    closed_duration = time.time() - (closed_start_time or time.time())
                    if (frame_counter >= MIN_CLOSED_FRAMES and
                        saw_fast_drop and
                        MIN_BLINK_DURATION <= closed_duration <= MAX_BLINK_DURATION):
                        blink_count += 1
                    # reset closure state
                    is_eye_closed = False
                    closed_start_time = None
                    saw_fast_drop = False
                    frame_counter = 0

            prev_smoothed_ear = smoothed_ear

        for point in left_eye_points + right_eye_points:
            cv2.circle(frame, point, 1, (0, 255, 255), -1)

        # helpful overlay for tuning
        cv2.putText(frame, f"EAR(sm): {smoothed_ear:.3f}" if smoothed_ear is not None else "EAR(sm): -", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Thr: {threshold:.3f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if movement_flag:
            cv2.putText(frame, "MOVEMENT - ignoring", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        # no face detected: reset transient states to avoid phantom blinks
        is_eye_closed = False
        closed_start_time = None
        saw_fast_drop = False
        frame_counter = 0
        prev_smoothed_ear = None

    # compute current BPM (same as you had)
    elapsed_time = time.time() - start_time
    bpm = blink_count / (elapsed_time / 60.0) if elapsed_time > 0 else 0.0

    # Flash logic (kept, slight change: use last_flash_time guard)
    current_time = time.time()
    if elapsed_time > 5:
        if bpm < 12 and (current_time - last_flash_time > 10):
            try:
                flash.flash_screen(color="white", duration=100, opacity=1)
            except Exception:
                # safe fallback if flash module fails on some systems
                pass
            last_flash_time = current_time

    cv2.imshow("Blink Detection (improved)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
