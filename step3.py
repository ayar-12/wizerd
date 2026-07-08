import time
import urllib.request
import os
from collections import deque

import cv2
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
        cv2.circle(frame, (x, y), 8, (0, 120, 255), -1)  
    else:
        trail.clear() 

    points = list(trail)
    for i in range(1, len(points)):
        thickness = max(1, int(6 * (i / len(points))))
        cv2.line(frame, points[i - 1], points[i], (255, 200, 60), thickness)

    cv2.imshow("Step 3 — fingertip trail", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()