import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

glass = cv2.imread('deal_with_it.png')

mask_positions = []

while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) >= 2:
        mask_positions.clear()

    for i in range(len(eyes)):
        (x, y, w, h) = eyes[i]

        closest_eye = None
        closest_eye_dist = 100000

        for j in range(i + 1, len(eyes)):
            dist = np.sqrt((x - eyes[j][0]) ** 2 + (y - eyes[j][1]) ** 2)
            if closest_eye is None or closest_eye_dist > dist > 70:
                closest_eye_dist = dist
                closest_eye = eyes[j]

        if closest_eye is not None:
            x = min(x, closest_eye[0])
            y = min(y, closest_eye[1])
            w = max(x + w, closest_eye[0] + closest_eye[2]) - x
            h = max(y + h, closest_eye[1] + closest_eye[3]) - y

            mask_positions.append((x, y, w, h))

    for mask_position in mask_positions:
        (x, y, w, h) = mask_position

        glass = cv2.resize(glass, (w, h), interpolation=cv2.INTER_AREA)

        mask = np.zeros_like(glass)

        mask[glass.mean(2) != 0] = 255

        glass = cv2.bitwise_and(glass, mask)

        cv2.add(frame[y:y + h, x:x + w], glass, frame[y:y + h, x:x + w])

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
