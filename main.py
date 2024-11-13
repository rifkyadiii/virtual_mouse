import cv2
import mediapipe as mp
import pyautogui
import random
import os
import numpy as np
import platform
import subprocess
import ctypes
import time

# Check operating system
is_windows = platform.system() == 'Windows'

# Screen resolution function
def get_screen_resolution():
    try:
        if is_windows:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            p = subprocess.Popen(['xrandr'], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(['grep', '*'], stdin=p.stdout, stdout=subprocess.PIPE)
            p.stdout.close()
            resolution = p2.communicate()[0].decode().split()[0]
            return map(int, resolution.split('x'))
    except Exception as e:
        print(f"Error: {e}. Default resolution applied.")
        return 1920, 1080

# Webcam initialization function
def initialize_webcam(width=1280, height=720):
    cap = cv2.VideoCapture(0 if is_windows else '/dev/video0')
    if not cap.isOpened():
        raise Exception("Webcam not accessible")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# Create centered window function
def create_centered_window(window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    screen_width, screen_height = get_screen_resolution()
    x, y = (screen_width - width) // 2, (screen_height - height) // 2
    cv2.moveWindow(window_name, x, y)
    cv2.resizeWindow(window_name, width, height)

# Process frame function
def process_frame(frame, width, height):
    frame = cv2.flip(cv2.resize(frame, (width, height)), 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.putText(frame, "Press 'q' to quit", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

# Gesture detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Utility functions for gesture detection
def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return np.abs(np.degrees(radians))

def get_distance(points):
    if len(points) < 2:
        return 0
    (x1, y1), (x2, y2) = points[0], points[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

# Tambahkan variabel untuk menyimpan posisi kursor sebelumnya
previous_x, previous_y = None, None
smoothing_factor = 0.3  # Turunkan smoothing factor untuk gerakan lebih halus
min_movement = 5  # Threshold minimal pergerakan untuk mengurangi getaran

def move_mouse(tip_coords, screen_width, screen_height):
    global previous_x, previous_y
    
    if tip_coords is not None:
        # Periksa apakah tip_coords memiliki atribut x dan y
        if hasattr(tip_coords, 'x') and hasattr(tip_coords, 'y'):
            x = int(tip_coords.x * screen_width)
            y = int(tip_coords.y / 2 * screen_height)
        else:
            x = int(tip_coords[0] * screen_width)
            y = int(tip_coords[1] / 2 * screen_height)

        # Perhalus gerakan kursor
        if previous_x is not None and previous_y is not None:
            # Hitung perubahan posisi
            dx = x - previous_x
            dy = y - previous_y
            
            # Terapkan threshold minimal pergerakan
            if abs(dx) < min_movement and abs(dy) < min_movement:
                return
                
            # Gunakan EMA dengan smoothing factor yang lebih rendah    
            smooth_x = int(smoothing_factor * x + (1 - smoothing_factor) * previous_x)
            smooth_y = int(smoothing_factor * y + (1 - smoothing_factor) * previous_y)
            
            # Gunakan moveRel dengan duration untuk gerakan lebih halus
            try:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.1, _pause=False)
            except:
                pass  # Abaikan error jika koordinat di luar layar
        else:
            # Untuk gerakan pertama
            pyautogui.moveTo(x, y, duration=0.1, _pause=False)
        
        # Update posisi sebelumnya
        previous_x, previous_y = x, y

def detect_gesture(frame, landmark_list, thumb_index_dist, screen_width, screen_height):
    if len(landmark_list) >= 21:
        gestures = {
            "left_click": get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and thumb_index_dist > 50,
            "right_click": get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and thumb_index_dist > 50,
            "double_click": get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and thumb_index_dist > 50,
            "screenshot": get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and thumb_index_dist < 50
        }

        # Gerakkan mouse hanya jika jarak antara ibu jari dan telunjuk di bawah threshold
        if thumb_index_dist < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            index_finger_tip = landmark_list[8]
            move_mouse(index_finger_tip, screen_width, screen_height)

        for gesture, condition in gestures.items():
            if condition:
                if gesture == "left_click":
                    pyautogui.click(button='left')
                elif gesture == "right_click":
                    pyautogui.click(button='right')
                elif gesture == "double_click":
                    pyautogui.doubleClick()
                elif gesture == "screenshot" and time.time() - detect_gesture.last_screenshot > 1:
                    screenshot = pyautogui.screenshot()
                    screenshot.save(f'screenshot_{random.randint(1, 1000)}.png')
                    detect_gesture.last_screenshot = time.time()
                    cv2.putText(frame, "Screenshot Taken", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, gesture.replace('_', ' ').title(), (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

detect_gesture.last_screenshot = 0

def main():
    width, height = 1280, 720
    WINDOW_NAME = "Virtual Mouse Control"
    screen_width, screen_height = get_screen_resolution()
    cap = initialize_webcam(width, height)
    create_centered_window(WINDOW_NAME, width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, width, height)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

            # Panggil move_mouse dengan objek landmark
            if thumb_index_dist < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
                move_mouse(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], screen_width, screen_height)

            detect_gesture(frame, landmark_list, thumb_index_dist, screen_width, screen_height)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
