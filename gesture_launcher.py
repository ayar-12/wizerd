import time
import math
import urllib.request
import os
import random
import webbrowser
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def get_app_support_dir():
    home = os.path.expanduser("~")
    dir_path = os.path.join(home, "Library", "Application Support", "GestureLauncher")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


MODEL_PATH = os.path.join(get_app_support_dir(), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model (one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

INDEX_FINGERTIP = 8
TRAIL_LENGTH = 40
trail = deque(maxlen=TRAIL_LENGTH)

CORE_COLOR = (60, 200, 255)
HOT_COLOR = (240, 245, 255)
TRIGGER_FLASH_COLOR = (255, 255, 255)

MIN_SPREAD_PX = 80
CIRCLE_ROTATION_THRESHOLD = 5.8
CLOSE_LOOP_RATIO = 0.25
MIN_SWIPE_DIST = 220
COOLDOWN_SECONDS = 3.0
GMAIL_URL = "https://mail.google.com"
PALMBEAUTY_URL = "https://www.palmbeauty.net"
YOURMONSTER_URL = "https://yourmonster.net"


def detect_checkmark(points):
    if len(points) < TRAIL_LENGTH:
        return False

    valley_idx = max(range(len(points)), key=lambda i: points[i][1])
    if valley_idx < int(len(points) * 0.15) or valley_idx > int(len(points) * 0.65):
        return False

    start, valley, end = points[0], points[valley_idx], points[-1]
    first_len = math.hypot(valley[0] - start[0], valley[1] - start[1])
    second_len = math.hypot(end[0] - valley[0], end[1] - valley[1])
    if first_len < 40 or second_len < 40:
        return False

    net_dx = end[0] - start[0]
    net_dy = end[1] - start[1]
    return second_len > first_len * 1.4 and net_dx > 40 and net_dy < 0


def detect_zigzag(points):
    if len(points) < TRAIL_LENGTH:
        return False

    net_dx = points[-1][0] - points[0][0]
    if net_dx < 150:
        return False

    step = 4
    sampled = points[::step]
    if len(sampled) < 4:
        return False

    direction_changes = 0
    prev_sign = None
    for i in range(1, len(sampled)):
        dy = sampled[i][1] - sampled[i - 1][1]
        if abs(dy) < 20:
            continue
        sign = 1 if dy > 0 else -1
        if prev_sign is not None and sign != prev_sign:
            direction_changes += 1
        prev_sign = sign

    return direction_changes >= 3


def detect_circle(points):
    if len(points) < TRAIL_LENGTH:
        return False

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    if spread < MIN_SPREAD_PX:
        return False

    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    angles = [math.atan2(y - cy, x - cx) for x, y in points]

    total_rotation = 0.0
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        total_rotation += diff

    closing_distance = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    return abs(total_rotation) > CIRCLE_ROTATION_THRESHOLD and closing_distance < spread * CLOSE_LOOP_RATIO


def detect_swipe(points):
    if len(points) < TRAIL_LENGTH:
        return False

    start, end = points[0], points[-1]
    straight_dist = math.hypot(end[0] - start[0], end[1] - start[1])
    if straight_dist < MIN_SWIPE_DIST:
        return False

    path_length = sum(
        math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
        for i in range(1, len(points))
    )
    if path_length == 0:
        return False

    straightness = straight_dist / path_length
    return straightness > 0.92


def detect_thumbs_up(landmarks):
    wrist = landmarks[0]
    thumb_mcp, thumb_tip = landmarks[2], landmarks[4]

    thumb_up = thumb_tip.y < thumb_mcp.y - 0.08 and thumb_tip.y < wrist.y - 0.12

    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    curled = all(landmarks[tip].y > landmarks[pip].y + 0.02 for tip, pip in finger_pairs)

    return thumb_up and curled


def hand_center_px(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def detect_clap(hand_landmarks_list, w, h):
    if len(hand_landmarks_list) != 2:
        return None
    c1 = hand_center_px(hand_landmarks_list[0], w, h)
    c2 = hand_center_px(hand_landmarks_list[1], w, h)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


cap = cv2.VideoCapture(0)
start_time = time.time()
last_trigger_time = 0.0
flash_until = 0.0
paused = False
last_stop_time = 0.0
last_resume_time = 0.0
ACTION_COOLDOWN_SECONDS = 1.5
thumbs_up_streak = 0
THUMBS_UP_STREAK_NEEDED = 8
hands_were_apart = True
CLAP_CLOSE_PX = 100
CLAP_APART_PX = 220
status_message = "Circle=Gmail | Checkmark=PALM Beauty | Zigzag=Your Monster | Swipe=Gmail | Clap=Play | Thumbs-up=Stop"
status_until = time.time() + 4

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    now = time.time()
    hands = result.hand_landmarks

    clap_distance = detect_clap(hands, w, h)
    if clap_distance is not None:
        if clap_distance > CLAP_APART_PX:
            hands_were_apart = True
        elif (clap_distance < CLAP_CLOSE_PX and hands_were_apart
              and now - last_resume_time > ACTION_COOLDOWN_SECONDS):
            paused = False
            hands_were_apart = False
            last_resume_time = now
            flash_until = now + 0.25
            status_message = "Clap detected -- resumed"
            status_until = now + 3

    if hands:
        landmarks = hands[0]

        if detect_thumbs_up(landmarks):
            thumbs_up_streak += 1
        else:
            thumbs_up_streak = 0

        if (thumbs_up_streak >= THUMBS_UP_STREAK_NEEDED
                and now - last_stop_time > ACTION_COOLDOWN_SECONDS):
            paused = True
            last_stop_time = now
            thumbs_up_streak = 0
            trail.clear()
            flash_until = now + 0.25
            status_message = "Thumbs-up detected -- stopped (clap to resume)"
            status_until = now + 3

        if not paused:
            tip = landmarks[INDEX_FINGERTIP]
            x, y = int(tip.x * w), int(tip.y * h)
            trail.append((x, y))
    else:
        trail.clear()
        thumbs_up_streak = 0

    points = [] if paused else list(trail)

    if not paused and now - last_trigger_time > COOLDOWN_SECONDS:
        fired = None
        if detect_circle(points):
            fired = ("Circle", GMAIL_URL, "Circle detected -- opening Gmail")
        elif detect_checkmark(points):
            fired = ("Checkmark", PALMBEAUTY_URL, "Checkmark detected -- opening PALM Beauty")
        elif detect_zigzag(points):
            fired = ("Zigzag", YOURMONSTER_URL, "Zigzag detected -- opening Your Monster")
        elif detect_swipe(points):
            fired = ("Swipe", GMAIL_URL, "Swipe detected -- opening Gmail")

        if fired:
            _, url, message = fired
            webbrowser.open(url)
            last_trigger_time = now
            flash_until = now + 0.25
            status_message = message
            status_until = now + 3
            trail.clear()
            points = []

    glow_layer = np.zeros_like(frame, dtype=np.uint8)
    for i in range(1, len(points)):
        t = i / max(1, len(points) - 1)
        color = tuple(int(CORE_COLOR[c] + (HOT_COLOR[c] - CORE_COLOR[c]) * t) for c in range(3))
        thickness = max(1, int(2 + 6 * t))
        cv2.line(glow_layer, points[i - 1], points[i], color, thickness, lineType=cv2.LINE_AA)

    if points:
        blurred = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=9, sigmaY=9)
        frame = cv2.add(frame, blurred)
        for i in range(1, len(points)):
            t = i / max(1, len(points) - 1)
            color = tuple(int(CORE_COLOR[c] + (HOT_COLOR[c] - CORE_COLOR[c]) * t) for c in range(3))
            cv2.line(frame, points[i - 1], points[i], color, 2, lineType=cv2.LINE_AA)

    for (px, py) in points[-15:]:
        if random.random() < 0.35:
            ox, oy = px + random.randint(-10, 10), py + random.randint(-10, 10)
            radius = random.choice([1, 1, 2])
            brightness = random.randint(180, 255)
            cv2.circle(frame, (ox, oy), radius, (brightness, brightness, 255), -1, lineType=cv2.LINE_AA)

    if now < flash_until:
        flash = np.full_like(frame, 255, dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.6, flash, 0.4, 0)

    if now < status_until:
        cv2.putText(frame, status_message, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED", (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Launcher", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
