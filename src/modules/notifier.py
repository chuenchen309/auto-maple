"""A module for detecting and notifying the user of dangerous in-game events."""

from src.common import config, utils
import time
import os
import cv2
import pygame
import threading
import numpy as np
import keyboard as kb
from src.routine.components import Point


# A rune's symbol on the minimap
RUNE_RANGES = (
    ((141, 148, 245), (146, 158, 255)),
)
rune_filtered = utils.filter_color(cv2.imread('assets/rune_template.png'), RUNE_RANGES)
RUNE_TEMPLATE = cv2.cvtColor(rune_filtered, cv2.COLOR_BGR2GRAY)

# Other players' symbols on the minimap
OTHER_RANGES = (
    ((0, 245, 215), (10, 255, 255)),
)
other_filtered = utils.filter_color(cv2.imread('assets/other_template.png'), OTHER_RANGES)
OTHER_TEMPLATE = cv2.cvtColor(other_filtered, cv2.COLOR_BGR2GRAY)

# The Elite Boss's warning sign
ELITE_TEMPLATE = cv2.imread('assets/elite_template.jpg', 0)

# 新版符文檢測用的模板
rune_main_filtered = cv2.imread('assets/rune_main_template.png')
RUNE_MAIN_TEMPLATE = cv2.cvtColor(rune_main_filtered, cv2.COLOR_BGR2GRAY)
# RUNE_MAIN_TEMPLATE = cv2.cvtColor(rune_main_filtered, cv2.COLOR_BGR2GRAY)

# 載入所有需要的���圖片
# RUNE_MAIN_TEMPLATE = cv2.imread('assets/rune_main_template.png')
# CHARACTER_MAIN_TEMPLATE = cv2.imread('assets/player_main_template.png')
RUNE_MAIN_TEMPLATE = cv2.cvtColor(cv2.imread('assets/rune_main_template.png'), cv2.COLOR_BGR2GRAY)
CHARACTER_MAIN_TEMPLATE = cv2.cvtColor(cv2.imread('assets/player_main_template.png'), cv2.COLOR_BGR2GRAY)

def get_alert_path(name):
    return os.path.join(Notifier.ALERTS_DIR, f'{name}.mp3')


class Notifier:
    ALERTS_DIR = os.path.join('assets', 'alerts')
    minimap_scale = 13.0  # 固定縮放比例為 13 倍

    def __init__(self):
        """Initializes this Notifier object's main thread."""

        pygame.mixer.init()
        self.mixer = pygame.mixer.music

        self.ready = False
        self.thread = threading.Thread(target=self._main)
        self.thread.daemon = True

        self.room_change_threshold = 0.9
        self.rune_alert_delay = 270         # 4.5 minutes

    def start(self):
        """Starts this Notifier's thread."""

        print('\n[~] Started notifier')
        self.thread.start()

    def _main(self):
        self.ready = True
        prev_others = 0
        rune_start_time = time.time()

        while True:
            if config.enabled:
                frame = config.capture.frame

                # Check for unexpected black screen
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.count_nonzero(gray < 15) / frame.shape[0] / frame.shape[1] > self.room_change_threshold:
                    self._alert('siren')

                # Check for elite warning
                elite_frame = frame[frame.shape[0] // 4:3 * frame.shape[0] // 4, frame.shape[1] // 4:3 * frame.shape[1] // 4]
                elite = utils.multi_match(elite_frame, ELITE_TEMPLATE, threshold=0.9)
                if len(elite) > 0:
                    self._alert('siren')

                # Check for other players entering the map
                filtered = utils.filter_color(config.capture.minimap['minimap'], OTHER_RANGES)
                others = len(utils.multi_match(filtered, OTHER_TEMPLATE, threshold=0.5))
                config.stage_fright = others > 0
                if others != prev_others:
                    if others > prev_others:
                        self._ping('ding')
                    prev_others = others

                # Check for rune
                now = time.time()
                if not config.bot.rune_active:
                    frame = config.capture.frame
                    minimap = config.capture.minimap['minimap']

                    # 使用新的計算邏輯
                    found, rune_minimap_pos = calculate_rune_minimap_position(frame, minimap)
                    if found:
                        rune_start_time = now
                        if config.routine.sequence:
                            # 使用計算出的小地圖位置，而不是固定的中心點
                            config.bot.rune_pos = rune_minimap_pos
                            distances = list(map(distance_to_rune, config.routine.sequence))
                            index = np.argmin(distances)
                            config.bot.rune_closest_pos = config.routine[index].location
                            config.bot.rune_active = True
                            self._ping('rune_appeared', volume=0.75)
                            print(f"[~] 地圖輪小地圖位置: {rune_minimap_pos}")
                            print(f"[~] 最近的路徑點: {config.bot.rune_closest_pos}")

                elif now - rune_start_time > self.rune_alert_delay:     # Alert if rune hasn't been solved
                    config.bot.rune_active = False
                    self._alert('siren')
            time.sleep(0.05)

    def _alert(self, name, volume=0.75):
        """
        Plays an alert to notify user of a dangerous event. Stops the alert
        once the key bound to 'Start/stop' is pressed.
        """

        config.enabled = False
        config.listener.enabled = False
        self.mixer.load(get_alert_path(name))
        self.mixer.set_volume(volume)
        self.mixer.play(-1)
        while not kb.is_pressed(config.listener.config['Start/stop']):
            time.sleep(0.1)
        self.mixer.stop()
        time.sleep(2)
        config.listener.enabled = True

    def _ping(self, name, volume=0.5):
        """A quick notification for non-dangerous events."""

        self.mixer.load(get_alert_path(name))
        self.mixer.set_volume(volume)
        self.mixer.play()

#################################
#       Helper Functions        #
#################################
def distance_to_rune(point):
    """
    Calculates the distance from POINT to the rune.
    :param point:   The position to check.
    :return:        The distance from POINT to the rune, infinity if it is not a Point object.
    """

    if isinstance(point, Point):
        return utils.distance(config.bot.rune_pos, point.location)
    return float('inf')

def detect_rune_in_main_screen(frame, minimap):
    """使用模板匹配檢測主畫面中的符文"""
    try:
        debug_image = frame.copy()
        matches = utils.multi_match(frame, RUNE_MAIN_TEMPLATE, threshold=0.6)

        if len(matches) > 0:
            # 取得���置並轉換為整數座標
            rune_pos = matches[0]
            center = (int(rune_pos[0]), int(rune_pos[1]))  # 確保座標是整數
            print(f"\n[!] 發現地圖輪! 位置: {center}")

            # 在圖片上標記地圖輪位置
            cv2.circle(debug_image, center, 20, (0, 255, 0), 2)  # 綠色圓圈
            cv2.putText(debug_image, f"Rune {center}",
                       (center[0]-10, center[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

            # 保存主畫面檢測結果
            cv2.imwrite('debug_rune_detection.jpg', debug_image)

            # 轉換為相對位置並視覺化
            # self.frame = self.screenshot()
            # if self.frame is None:
            #     continue

            # # Crop the frame to only show the minimap
            # minimap = self.frame[mm_tl[1]:mm_br[1], mm_tl[0]:mm_br[0]]
            rune_pos = utils.convert_to_relative(center, minimap)

            # 在小地圖上標記位置
            minimap_debug = minimap.copy()
            minimap_pos = (int(rune_pos[0] * minimap.shape[1]),
                         int(rune_pos[1] * minimap.shape[0]))
            cv2.circle(minimap_debug, minimap_pos, 5, (0, 255, 0), 2)
            cv2.putText(minimap_debug, f"Rune {rune_pos}",
                       (minimap_pos[0]-10, minimap_pos[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.3, (0, 255, 0), 1)

            # 保存小地圖檢測結果
            cv2.imwrite('debug_rune_minimap.jpg', minimap_debug)

            return True, rune_pos

        return False, None

    except Exception as e:
        print(f"\n[!] 檢測地圖輪時發生錯誤: {str(e)}")
        return False, None

def detect_character_in_main_screen(frame):
    """檢測主畫面中角色的位置"""
    try:
        # 複製原始圖片，用於繪製結果
        # debug_image = frame.copy()

        # 使用模板匹配找到角色位置
        matches = utils.multi_match(frame, CHARACTER_MAIN_TEMPLATE, threshold=0.6)

        if len(matches) > 0:
            pos = matches[0]
            print(f"\n[!] 發現角色! 位置: {pos}")

            # 在圖片上標記角色位置
            # cv2.circle(debug_image, pos, 20, (0, 0, 255), 2)  # 紅色圓圈
            # cv2.putText(debug_image, f"Character {pos}",
            #            (pos[0]-10, pos[1]-10),
            #            cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, (0, 0, 255), 2)

            # # 保存結果圖片
            # cv2.imwrite('debug_character_detection.jpg', debug_image)
            return True, pos

        return False, None

    except Exception as e:
        print(f"\n[!] 檢測角色時發生錯誤: {str(e)}")
        return False, None

def calculate_rune_minimap_position(frame, minimap):
    """計算地圖輪在小地圖上的位置"""
    try:
        # 1. 檢測主畫面中的符文和角色位置
        rune_found, rune_pos = detect_rune_in_main_screen(frame, minimap)
        character_found, char_main_pos = detect_character_in_main_screen(frame)
        if not rune_found:
            return False, None


        # 2. 計算主畫面中的相對距離差
        # main_dx = rune_main_pos[0] - char_main_pos[0]  # 主畫面X軸距離差
        # main_dy = rune_main_pos[1] - char_main_pos[1]  # 主畫面Y軸距離差

        # print(f"\n[!] 主畫面距離差: dx={main_dx:.2f}, dy={main_dy:.2f}")


        # 4. 將主畫面距離差轉換為小地圖距離差
        # point = (main_dx, main_dy)
        # mini_dx, mini_dy = utils.convert_to_relative(point, minimap)
        # mini_dx2, mini_dy2 = utils.convert_to_relative(point, frame)
        # print(f"\n[!] 小地圖距離差: dx={mini_dx:.2f}, dy={mini_dy:.2f}")
        # print(f"[!] 小地圖距離差2: dx={mini_dx2:.2f}, dy={mini_dy2:.2f}")

        # # 5. 從小地圖角色位置計算符文位置
        # char_minimap_pos = config.player_pos
        # rune_minimap_x = char_minimap_pos[0] + mini_dx
        # rune_minimap_y = char_minimap_pos[1] + mini_dy

        print(f"\n[!] 小地圖角色位置: {config.player_pos}")
        print(f"[!] 計算出的小地圖符文位置: {rune_pos}")
        return True, rune_pos



    except Exception as e:
        print(f"\n[!] 計算符文位置時發生錯誤: {str(e)}")
        return False, None