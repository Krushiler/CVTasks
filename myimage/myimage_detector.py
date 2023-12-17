import sys

import cv2

my_image = cv2.imread('img.png')
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
my_image = cv2.threshold(my_image, 10, 255, cv2.THRESH_BINARY)[1]

found_frames = []

def find_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, 0)
    result = cv2.matchTemplate(thresh, my_image, cv2.TM_CCOEFF_NORMED)
    return result > 0.9


def read_video():
    cap = cv2.VideoCapture('output.avi')
    frame_index = 0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while not cap.isOpened():
        cap = cv2.VideoCapture('output.avi')
        cv2.waitKey(1000)

    while cap.isOpened():
        flag, frame = cap.read()
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if flag:
            result = find_contours(frame)
            frame_index += 1
            if result:
                found_frames.append(frame_index)
                print(f'Frame: {frame_index}')
        if current_frame >= total_frames:
            break


read_video()

print('Found frames: ', len(found_frames))
