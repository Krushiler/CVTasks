import socket

import numpy as np
import cv2
from scipy.signal import argrelextrema
from skimage.measure import regionprops, label
from math import dist, sqrt

WINDOW_SCREEN = 'screenshot'
IMAGE_SIZE = 40002

cv2.namedWindow(WINDOW_SCREEN, cv2.WINDOW_NORMAL)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def find_extremes(image):
    extremes = []
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            current_value = image[i, j]
            if (image[i, j + 1] < current_value and image[i, j - 1] < current_value
                    and image[i + 1, j] < current_value and image[i - 1, j] < current_value):
                extremes.append((i, j))
    return extremes


host = "84.237.21.36"
port = 5152

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    beat = b"nope"
    while True:
        sock.send(b"get")
        bts = recvall(sock, IMAGE_SIZE)
        print(len(bts))

        im1 = np.frombuffer(bts[2:IMAGE_SIZE], dtype="uint8").reshape(bts[0], bts[1])
        pos1 = np.unravel_index(im1.argmax(), im1.shape)
        res = np.abs(np.array(pos1))

        sock.send(f"{res[0]}".encode())
        print(sock.recv(4))

        extremes = find_extremes(im1)

        distance = sqrt((extremes[0][0] - extremes[1][0]) ** 2 + (extremes[0][1] - extremes[1][1]) ** 2)

        distance = round(distance * 10) / 10

        cv2.imshow(WINDOW_SCREEN, im1)

        result = str(distance).encode()

        cv2.waitKey(0)

        sock.send(result)
        beat = sock.recv(20)
