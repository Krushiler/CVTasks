import numpy as np
import zmq
import cv2

# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.connect("tcp://192.168.0.105:6556")
# socket.subscribe(b'')
#

frame_t = [[]]


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(frame_t[y, x])


cv2.namedWindow('frame')
cv2.namedWindow('binary')

cv2.setMouseCallback('binary', on_mouse_click)

rects = []
circles = []


def analyze_image(frame):
    global frame_t
    frame_t = frame

    frame_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2LUV)

    L, A, B = cv2.split(frame_t)

    frame_t = B

    frame_t = cv2.medianBlur(frame_t, 15)

    frame_t = cv2.Canny(frame_t, 30, 10)

    frame_t = cv2.dilate(frame_t, None, iterations=1)

    cv2.imshow('binary', frame_t)

    frame_t_c, hierarchy = cv2.findContours(frame_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = []

    for contour in frame_t_c:
        if cv2.contourArea(contour) > 200:
            contours.append(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            (x, y), rad = cv2.minEnclosingCircle(contour)
            center = int(x), int(y)
            rad = int(rad)
            cv2.circle(frame, center, rad, (0, 255, 0), 2)

            rect_area = rect[1][0] * rect[1][1]
            circle_area = np.pi * rad ** 2

            if rect_area < circle_area:
                rects.append(rect)
            else:
                circles.append((center, rad))

    print(f'rects: {len(rects)}')
    print(f'circles: {len(circles)}')
    print(f'total: {len(rects) + len(circles)}')

    cv2.imshow('frame', frame)
    cv2.waitKey(0)


prev_frame = None

frame = cv2.imread('img.jpg')
analyze_image(frame)
