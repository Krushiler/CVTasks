from enum import Enum

import cv2
import numpy as np
from skimage.measure import label


class FigureType(Enum):
    RECT = 0
    CIRCLE = 1


def print_result_row(name, result):
    print("{:<20} {:<20}".format(name, result))


def count_elements(labeled, img):
    results = {}
    rect_count = 0
    circle_count = 0

    for box_index in range(1, np.max(labeled) + 1):
        box = np.where(labeled == box_index)

        y_min, x_min = box[0][0], box[1][0]
        y_max, x_max = box[0][-1], box[1][-1]

        box_area = (x_max - x_min + 1) * (y_max - y_min + 1)

        figure_type = FigureType.RECT if box_area == len(box[0]) else FigureType.CIRCLE

        figure_shade = img[y_min, x_min, 0]

        results.setdefault(figure_shade, [0, 0])

        results[figure_shade][figure_type.value] += 1

        if figure_type == FigureType.RECT:
            rect_count += 1
        elif figure_type == FigureType.CIRCLE:
            circle_count += 1

    return results, rect_count, circle_count


image_path = "balls_and_rects.png"

image = cv2.imread(image_path)
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, img_gray = cv2.threshold(img_gray, 128, 192, cv2.THRESH_OTSU)

labeled = label(img_gray)

results, rect_count, circle_count = count_elements(labeled, img_hsv)

print_result_row("Фигур:", np.max(labeled))
print_result_row("Цветов:", len(results))
print_result_row("Прямоугольников:", rect_count)
print_result_row("Кругов:", circle_count)

print()

print("{:<20} {:<20} {:<20} {:<20}".format("Оттенок", "Прямоугольники", "Круги", "Всего"))

for shade, counters in results.items():
    print("{:<20} {:<20} {:<20} {:<20}".format(shade, counters[0], counters[1], counters[0] + counters[1]))
