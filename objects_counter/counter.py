import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import *


def crop_figure(img, fig):
    indices = np.where(img == fig)
    y_max, y_min = np.max(indices[0]), np.min(indices[0])
    x_max, x_min = np.max(indices[1]), np.min(indices[1])
    return img[y_min:y_max + 1, x_min:x_max + 1] // fig


image = np.load("ps.npy")

labeled_image, figures_count = label(image)

unique_figures = []
unique_figures_counter = {}

for figure in range(1, figures_count + 1):
    unique = True
    component = crop_figure(labeled_image, figure)

    for index, current_figure in enumerate(unique_figures):
        if current_figure.shape == component.shape and np.equal(current_figure, component).all():
            if index not in unique_figures_counter:
                unique_figures_counter[index] = 1
            else:
                unique_figures_counter[index] += 1
            unique = False
            break

    if unique and np.any(component):
        unique_figures.append(component)
        unique_figures_counter[figure] = 1

for index, figure in enumerate(unique_figures):
    print(f"Элемент {index}: {unique_figures_counter[index]}")

plt.plot(image)
plt.show()
