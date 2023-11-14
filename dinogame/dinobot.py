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
    offset = int((score // 200) * 5)
    return offset


def can_crouch_in_air(horizontal_offset):
    above_dino_position = dino_capturer.find_above_dino()
    dino_position = dino_capturer.find_on_dino_position()
    empty_above = np.all(above_dino_position == 0)
    empty_on_dino = np.all(dino_position == 0)
    print(empty_on_dino)
    return empty_above or empty_on_dino


def wait_until_can_jump(horizontal_offset):
    while not can_crouch_in_air(horizontal_offset):
        time.sleep(0.01)


def calculate_offset():
    current_time = time.time()
    score = time_diff_to_score(current_time)
    return map_score_to_offset(score), score


def calculate_crouch_delay(score):
    return max(0.3 - 0.3 * (score // 150 * 150) / 3000, 0.1)


def analyze_crouch(offset):
    if can_crouch_in_air(horizontal_offset=offset):
        if not keyboard.is_pressed('down'):
            keyboard.press('down')


def analyze_bottom_obstacle(offset, score):
    print(offset)
    bottom_obstacles_image = dino_capturer.find_bottom_obstacle(horizontal_offset=offset)
    if np.any(bottom_obstacles_image < 150):
        keyboard.release('down')
        keyboard.press('space')
        time.sleep(calculate_crouch_delay(score))
        keyboard.release('space')
        keyboard.press('down')


def play_game_step():
    offset, score = calculate_offset()
    analyze_bottom_obstacle(offset, score)
    # analyze_crouch(offset)
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
