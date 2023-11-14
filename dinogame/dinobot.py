import time

import cv2
import keyboard
import numpy as np

from dinogame.dino_capturer import DinoCapturer

dino_capturer = DinoCapturer()

start_time = time.time()

WINDOW_SCREEN = 'screenshot'


def time_diff_to_score(c_time):
    return (c_time - start_time) * 10


def map_score_to_offset(score):
    offset = int((score // 100) * 7.5)
    return offset


def calculate_offset():
    current_time = time.time()
    score = time_diff_to_score(current_time)
    return map_score_to_offset(score), score


def calculate_crouch_delay(score):
    if score < 500:
        return 0.3
    return max(0.3 - 0.3 * (score // 150 * 150) / 10000, 0.1)


def analyze_bottom_obstacle(offset, score):
    print(offset)
    bottom_obstacles_image = dino_capturer.find_bottom_obstacle(horizontal_offset=offset)
    if np.any(bottom_obstacles_image < 150):
        keyboard.release('down')
        keyboard.press('space')
        time.sleep(calculate_crouch_delay(score))
        keyboard.release('space')
        keyboard.press('down')
        time.sleep(0.15)


def play_game_step():
    offset, score = calculate_offset()
    analyze_bottom_obstacle(offset, score)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def find_dino_step():
    dino_capturer.find_dino_on_screen()


is_game_step = False
while True:
    if keyboard.is_pressed('enter'):
        is_game_step = not is_game_step
        time.sleep(0.5)
        start_time = time.time()
    if is_game_step:
        play_game_step()
    else:
        find_dino_step()
