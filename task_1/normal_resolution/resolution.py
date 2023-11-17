import numpy as np

for i in range(6):
    filename = f"data/figure{i+1}.txt"
    with open(filename, "r") as file:
        max_size_mm = float(file.readline())
        data = np.loadtxt(file, dtype=int)

    obj = [row for row in data if np.any(row > 0)]
    width_pixels = len(obj)

    if width_pixels == 0:
        print(f"{filename} object not found")
        continue

    nominal_resolution = max_size_mm / width_pixels

    print(f"{filename} {nominal_resolution}")
