import numpy as np

for i in range(6):
    filename = f"data/figure{i+1}.txt"
    with open(filename, "r") as file:   
        max_size_mm = float(file.readline())
        data = np.loadtxt(file, dtype=int)

    width_pixels = len(data[0])

    nominal_resolution = max_size_mm / width_pixels

    print(f"{filename} {nominal_resolution}")
