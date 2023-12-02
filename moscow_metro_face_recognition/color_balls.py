import random

import cv2


class BallColor:
    def __init__(self, lower, upper, debug_color=(0, 0, 255)):
        self.lower = lower
        self.upper = upper
        self.debug_color = debug_color


blue_color = BallColor((90, 100, 100), (110, 255, 255), (255, 0, 0))
red_color = BallColor((170, 100, 100), (190, 255, 255), (0, 0, 255))
yellow_color = BallColor((20, 100, 100), (40, 255, 255), (0, 255, 255))
green_color = BallColor((50, 100, 100), (90, 255, 255), (0, 255, 0))

colors = [blue_color, yellow_color, green_color]

random.shuffle(colors)


def find_ball(frame, color):
    mask = cv2.inRange(hsv, color.lower, color.upper)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, (int(x), int(y)), int(radius), color.debug_color, 2)
        cv2.circle(frame, center, 5, color.debug_color, -1)
        return center

    return None


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_EXPOSURE, -1)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

cv2.namedWindow("Camera")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_EXPOSURE, -1)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

is_win = False

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)

    balls = []

    for color in colors:
        ball = find_ball(frame, color)
        if ball is not None:
            balls.append(ball)

    if len(balls) == len(colors):
        prev_pos = None
        is_win = True
        for i, ball in enumerate(balls):
            if prev_pos is not None and balls[i][0] > prev_pos[0]:
                cv2.line(frame, prev_pos, ball, colors[i].debug_color, 2)
            elif i > 0:
                is_win = False
            prev_pos = ball

    if is_win:
        cv2.putText(frame, "WIN", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()
