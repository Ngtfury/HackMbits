import tkinter as tk
import time

def subtle_warm_flash(duration=6, max_alpha=0.1, steps=60):
    """
    Subtly warm the screen and restore it without being obvious.
    
    :param duration: Total duration of the flash in seconds
    :param max_alpha: Maximum opacity of the overlay (0-1), keep low for subtlety
    :param steps: Number of steps for smooth transition
    """
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0)  # start invisible
    root.configure(bg="#FFC87F")  # warm color

    # Gradually increase opacity (fade-in)
    for i in range(steps):
        alpha = max_alpha * (i + 1) / steps
        root.attributes('-alpha', alpha)
        root.update()
        time.sleep(duration / (4 * steps))

    # Gradually decrease opacity (fade-out)
    for i in range(steps):
        alpha = max_alpha * (1 - (i + 1) / steps)
        root.attributes('-alpha', alpha)
        root.update()
        time.sleep(duration / (4 * steps))

    root.destroy()

# Example usage
subtle_warm_flash()
