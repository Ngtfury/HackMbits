import tkinter as tk
import threading

def flash_screen(color="white", duration=1000, opacity=0.1):
    """Flashes a semi-transparent fullscreen overlay."""
    def show_flash():
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.configure(bg=color)
        root.attributes("-alpha", opacity)
        root.after(duration, root.destroy)
        root.mainloop()
    threading.Thread(target=show_flash, daemon=True).start()
