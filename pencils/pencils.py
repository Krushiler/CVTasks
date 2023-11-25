import os

import cv2
import numpy as np
from skimage import filters, measure


def find_pencils(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    binary_image = (image < filters.threshold_otsu(image)).astype(np.uint8)

    _, region = cv2.connectedComponents(binary_image)

    pencils = sum(
        region.perimeter > 1000 and 30 > (region.major_axis_length / region.minor_axis_length) > 15 for region in
        measure.regionprops(region)
    )

    return pencils


folder = 'images/'
image_files = os.listdir(folder)

pencils_count = 0

for file in image_files:
    count_pencils = find_pencils(f'{folder}{file}')
    pencils_count += count_pencils

print(pencils_count)
