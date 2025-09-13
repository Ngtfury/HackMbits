import threading
import time
import features.components.brightness as bright

class EyeDistance:
    def __init__(self):
        self.running = False
        self.thread = None

    def start(self):
        """Start the eye distance monitoring"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            print("[EyeDistance] Started")

    def stop(self):
        """Stop eye distance monitoring"""
        self.running = False
        print("[EyeDistance] Stopped")

    def run(self):
        """Put your eye distance logic here"""
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
        while self.running:
            print("[EyeDistance] Monitoring...")
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
            time.sleep(2)

        
