import cv2

# Constants
KNOWN_IPD_CM = 6.3  # Average interpupillary distance in cm
FOCAL_LENGTH = 700  # Approximate focal length in pixels (may need calibration)

# Load face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Could not access webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    distance_text = "No face"
    for (x, y, fw, fh) in faces:
        roi_gray = gray[y:y+fh, x:x+fw]
        roi_color = frame[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            # Take two largest eyes (by width)
            eyes = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                eye_centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)
            # Calculate pixel distance between eyes
            dx = eye_centers[0][0] - eye_centers[1][0]
            dy = eye_centers[0][1] - eye_centers[1][1]
            pixel_dist = (dx**2 + dy**2) ** 0.5
            # Estimate distance (D = (W * F) / P)
            distance_cm = round((KNOWN_IPD_CM * FOCAL_LENGTH) / pixel_dist, 1)
            distance_text = f"Distance: {distance_cm} cm"
        else:
            distance_text = "Eyes not detected"

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(frame, distance_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam - Eye Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()