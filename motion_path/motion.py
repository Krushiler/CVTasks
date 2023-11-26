import cv2
import matplotlib.pyplot as plt
import numpy as np


class Circle:
    def __init__(self, x, y):
        self.last_x = x
        self.last_y = y
        self.cords = [(self.last_x, self.last_y)]

    def is_cord_connected(self, new_cords, threshold=10):
        distance = ((new_cords[0] - self.last_x) ** 2 + (new_cords[1] - self.last_y) ** 2) ** 0.5
        return distance <= threshold

    def append_cord(self, cords):
        self.last_x = cords[0]
        self.last_y = cords[1]
        self.cords += [(self.last_x, self.last_y)]

    def x_trace(self):
        return [cord[0] for cord in self.cords]

    def y_trace(self):
        return [cord[1] for cord in self.cords]


def find_circle(file_path):
    cords = []

    img = np.load(file_path)
    circles, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for circle in circles:
        moments = cv2.moments(circle)
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        cords += [(center_x, center_y)]
    return cords


def process_circle(cords, ix_cords, circle):
    circle.append_cord(cords[ix_cords])
    cords.pop(ix_cords)


files = [f'out/h_{i}.npy' for i in range(100)]

c_cords = find_circle(files[0])

files = files[1:]

c1 = Circle(c_cords[0][0], c_cords[0][1])
c2 = Circle(c_cords[1][0], c_cords[1][1])
c3 = Circle(c_cords[2][0], c_cords[2][1])

for f in files:
    cords = find_circle(f)
    idx = 0
    threshold = 5

    while len(cords) > 0:
        idx = (idx + 1) % len(cords)
        threshold += 5

        if c1.is_cord_connected(cords[idx], threshold):
            process_circle(cords, idx, c1)
        elif c2.is_cord_connected(cords[idx], threshold):
            process_circle(cords, idx, c2)
        elif c3.is_cord_connected(cords[idx], threshold):
            process_circle(cords, idx, c3)

x1, y1 = c1.x_trace(), c1.y_trace()
x2, y2 = c2.x_trace(), c2.y_trace()
x3, y3 = c3.x_trace(), c3.y_trace()

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)

plt.show()
