import cv2
import numpy as np
from mss import mss

dino_path = 'assets/dino.png'


class DinoCapturer:
    def __init__(self):
        self.capturer = mss()
        self.monitor = self.capturer.monitors[0]
        self.dino_location = None
        self.dino_width = 0
        self.dino_height = 0

    def get_screenshot(self):
        screenshot = self.capturer.grab(self.monitor)
        return np.array(screenshot)

    def find_above_dino(self):
        monitor = {"top": self.dino_location[1] - self.dino_height,
                   "left": self.dino_location[0],
                   "width": self.dino_width,
                   "height": 20}

        screenshot = np.array(self.capturer.grab(monitor))
        screenshot = cv2.Canny(screenshot, 100, 200)
        return screenshot

    def find_on_dino_position(self):
        monitor = {"top": self.dino_location[1] + 15,
                   "left": self.dino_location[0],
                   "width": self.dino_width,
                   "height": 5}

        screenshot = np.array(self.capturer.grab(monitor))
        screenshot = cv2.Canny(screenshot, 100, 200)

        return screenshot

    def find_bottom_obstacle(self, horizontal_offset):
        monitor = {"top": self.dino_location[1] + 5,
                   "left": self.dino_location[0] + self.dino_width,
                   "width": self.dino_width + horizontal_offset,
                   "height": 25}

        screenshot = np.array(self.capturer.grab(monitor))

        return screenshot

    def find_dino_on_screen(self):
        dino_image = cv2.imread(dino_path, cv2.IMREAD_GRAYSCALE)
        dino_image = cv2.Canny(dino_image, 100, 200)
        screenshot_original = self.get_screenshot()
        screenshot = cv2.cvtColor(screenshot_original, cv2.COLOR_BGR2GRAY)
        screenshot = cv2.Canny(screenshot, 100, 200)
        w, h = dino_image.shape
        result = cv2.matchTemplate(screenshot, dino_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        self.dino_location = max_loc
        self.dino_width = w
        self.dino_height = h
        return max_loc, (max_loc[0] + w, max_loc[1] + h)
