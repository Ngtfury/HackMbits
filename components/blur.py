import tkinter as tk
from PIL import Image, ImageTk, ImageFilter
import pyautogui
import time

class BlurOverlay:
    def __init__(self, blur_radius=20, message="Too close!", font_size=25):
        self.blur_radius = blur_radius
        self.root = None
        self.tk_img = None
        self.message = message
        self.font_size = font_size

    def toggle(self):
        if self.root: 
            self.root.destroy()
            self.root = None
        else:
            self.show_blur()

    def show_blur(self):

        screenshot = pyautogui.screenshot()
        img = screenshot.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.tk_img = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(self.root, width=screen_width, height=screen_height, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        canvas.create_text(
            screen_width/2,
            screen_height/2,
            text=self.message,
            fill="yellow",
            font=("Arial", self.font_size, "bold")
        )

        self.root.update()


if __name__ == "__main__":
    overlay = BlurOverlay(blur_radius=20, message="Too close!", font_size=25)
    overlay.toggle()
    time.sleep(2)
    overlay.toggle()
