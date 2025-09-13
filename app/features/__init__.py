# features/__init__.py

from .eye_distance import EyeDistance
from .blink_detection import BlinkDetection
from .adaptive_brightness import AdaptiveBrightness
from .night_light import NightLight

# Optional: make it easy to list all features
__all__ = [
    "EyeDistance",
    "BlinkDetection",
    "AdaptiveBrightness",
    "NightLight"
]
