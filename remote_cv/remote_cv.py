import socket
from math import sqrt

import cv2
import numpy as np

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
                extremes.append((j, i))
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

        extremes = find_extremes(im1)

        if len(extremes) == 2:
            distance = round(sqrt((extremes[0][0] - extremes[1][0]) ** 2 + (extremes[0][1] - extremes[1][1]) ** 2), 1)
        else:
            distance = 0

        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
        for extreme in extremes:
            cv2.drawMarker(im1, extreme, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=1)

        result = str(distance).encode()

        print(distance)

        sock.send(result)
        beat = sock.recv(20)
        print(beat)

        cv2.imshow(WINDOW_SCREEN, im1)
        cv2.waitKey(0)
