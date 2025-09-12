import cv2
import numpy as np
import os

def get_average_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness

def get_screen_brightness():
    # Try reading brightness from /sys/class/backlight (Linux, X11/Wayland, laptops)
    try:
        backlight_dir = "/sys/class/backlight"
        if os.path.isdir(backlight_dir):
            for name in os.listdir(backlight_dir):
                path = os.path.join(backlight_dir, name)
                with open(os.path.join(path, "brightness")) as f:
                    brightness = int(f.read())
                with open(os.path.join(path, "max_brightness")) as f:
                    max_brightness = int(f.read())
                percent = (brightness / max_brightness) * 100
                return round(percent, 1)
    except Exception:
        pass
    return None

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Could not access webcam!")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        avg = get_average_brightness(frame)
        screen_brightness = get_screen_brightness()
        if screen_brightness is not None:
            print(f"Average brightness: {avg:.2f} | Screen brightness: {screen_brightness}%")
        else:
            print(f"Average brightness: {avg:.2f} | Screen brightness: Not available")

        cv2.imshow("Webcam - Brightness Overlay", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
