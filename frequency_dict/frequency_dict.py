import numpy as np
from skimage.measure import label, regionprops
import cv2


def get_regions_count(image):
    inverted_image = ~image
    labeled_image = label(inverted_image)
    regions = regionprops(labeled_image)

    def is_on_boundary(cords, shape):
        return any(y in (0, shape[0] - 1) or x in (0, shape[1] - 1) for y, x in cords)

    count_internal = sum(1 for region in regions if not is_on_boundary(region.coords, image.shape))
    count_external = len(regions) - count_internal

    return count_internal, count_external


def has_vertical_line(region):
    vertical_sum = np.sum(region.image, axis=0)
    return 1 in (vertical_sum // region.image.shape[0])


def recognize_closed_symbol(region, externals):
    if has_vertical_line(region):
        return "1"

    if externals == 2:
        return "/"

    image = region.image[2:-2, 2:-2]
    _, cut_externals = get_regions_count(image)

    if cut_externals == 4:
        return "X"

    if cut_externals == 5:
        center_y, center_x = np.array(image.shape) // 2
        return "*" if image[center_y, center_x] > 0 else "W"

    return "*" if externals == 5 else "W"


def recognize_symbol(region):
    if np.all(region.image):
        return "-"

    internals, externals = get_regions_count(region.image)

    if internals == 2:
        return "B" if has_vertical_line(region) else "8"

    if internals == 1:
        if externals == 3:
            return "A"
        elif externals == 2:
            center_y, center_x = np.array(region.image.shape) // 2
            return "P" if region.image[center_y, center_x] > 0 else "D"
        else:
            return "0"

    if internals == 0:
        return recognize_closed_symbol(region, externals)

    return None


symbols_path = "data/symbols.png"
symbols_image = cv2.imread(symbols_path)
symbols_image = cv2.cvtColor(symbols_image, cv2.COLOR_BGR2GRAY)
(thresh, symbols_image) = cv2.threshold(symbols_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

labeled_symbols = label(symbols_image)
regions = regionprops(labeled_symbols)

frequencies = {}

missed_symbols = 0

for region in regions:
    symbol = recognize_symbol(region)
    if symbol is not None:
        labeled_symbols[labeled_symbols == region.label] = 0
    else:
        missed_symbols += 1

    frequencies[symbol] = frequencies.get(symbol, 0) + 1
    cv2.rectangle(symbols_image, (region.bbox[0], region.bbox[2]), (region.bbox[1], region.bbox[3]), (255, 0, 0), 2)

accuracy = 1. - missed_symbols / sum(frequencies.values())

print(f"Detection rate: {accuracy}")
print(f"Frequencies: {frequencies}")
