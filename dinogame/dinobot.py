import time

import cv2
import keyboard
import numpy as np

from dinogame.dino_capturer import DinoCapturer

dino_capturer = DinoCapturer()

start_time = time.time()

WINDOW_SCREEN = 'screenshot'

# cv2.namedWindow(WINDOW_SCREEN, cv2.WINDOW_NORMAL)


def time_diff_to_score(c_time):
    return (c_time - start_time) // 10


def map_score_to_offset(score):
    return int(score // 100 + 20)


def can_crouch_in_air(horizontal_offset):
    above_dino_position = dino_capturer.find_above_dino()
    dino_position = dino_capturer.find_on_dino_position()
    # cv2.imshow(WINDOW_SCREEN, dino_position)
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
    return map_score_to_offset(score)


def analyze_crouch(offset):
    if can_crouch_in_air(horizontal_offset=offset):
        if not keyboard.is_pressed('down'):
            keyboard.press('down')


def analyze_bottom_obstacle(offset):
    bottom_obstacles_image = dino_capturer.find_bottom_obstacle(horizontal_offset=offset)

    if np.any(bottom_obstacles_image < 150):
        keyboard.release('down')
        time.sleep(0.01)
        keyboard.press('space')
        time.sleep(0.01)
        keyboard.release('space')
        time.sleep(0.3)
        keyboard.press('down')


def play_game_step():
    offset = calculate_offset()
    analyze_bottom_obstacle(offset)
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
