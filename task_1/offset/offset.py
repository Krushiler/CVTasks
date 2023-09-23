import numpy as np

with open("data/img1.txt", "r") as file:
    file.readline()
    file.readline()
    img1_data = np.loadtxt(file, dtype=int)

with open("data/img2.txt", "r") as file:
    file.readline()
    file.readline()
    img2_data = np.loadtxt(file, dtype=int)


def find_offset(img1, img2):
    corr = np.correlate(img1.ravel(), img2.ravel(), mode='full')

    max_corr_index = np.argmax(corr)

    y = max_corr_index // img2.shape[1]
    x = max_corr_index % img2.shape[1]

    return y - img1.shape[0] + 1, x - img1.shape[1] + 1


offset_y, offset_x = find_offset(img1_data, img2_data)

print(f"y: {offset_y}")
print(f"x: {offset_x}")
