import time
import urllib.request
import os
import random
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand tracking model (one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

INDEX_FINGERTIP = 8
trail = deque(maxlen=40)

CORE_COLOR = (60, 200, 255)     
HOT_COLOR = (240, 245, 255)   

cap = cv2.VideoCapture(0)
start_time = time.time()

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

    if result.hand_landmarks:
        tip = result.hand_landmarks[0][INDEX_FINGERTIP]
        x, y = int(tip.x * w), int(tip.y * h)
        trail.append((x, y))
    else:
        trail.clear()

    points = list(trail)

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

    cv2.imshow("Step 4 — magic trail", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()