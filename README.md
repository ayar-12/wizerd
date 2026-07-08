# Gesture Launcher

Control your computer with hand gestures in front of your webcam — draw a shape in the air, trigger a real action. Built with OpenCV + MediaPipe. No ML training, no Electron, just geometry.

![status](https://img.shields.io/badge/status-proof--of--concept-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

## What it does

Point your index finger at the camera and draw:

| Gesture | Action |
|---|---|
| Circle | Opens Gmail |
| Checkmark | Opens PALM Beauty |
| Zigzag | Opens Your Monster |
| Swipe | Opens Gmail |
| Thumbs-up (held) | Stops gesture detection |
| Clap (two hands) | Resumes gesture detection |

Every gesture is drawn as a glowing, gold-to-white trail with a soft blur bloom and sparkle particles — not just a flat line.

## How it works

1. **MediaPipe HandLandmarker** tracks 21 points on your hand in real time, from your webcam feed via OpenCV.
2. The last ~40 positions of your index fingertip are kept as a rolling trail.
3. Each gesture is a **pure geometry check** on that trail — no machine learning, no training data:
   - **Circle**: does the trail sweep close to a full 360 degrees around its own center, and end near where it started?
   - **Checkmark**: a short stroke down, then a longer stroke up-and-right?
   - **Zigzag**: net left-to-right travel with at least 3 vertical direction changes?
   - **Swipe**: is straight-line distance from start to end almost equal to the total path length (a fast direct line, not a curve)?
4. **Thumbs-up** and **clap** are separate, static/two-hand checks that stop and resume detection — useful so normal hand movement while talking doesn't accidentally trigger gestures.

## Setup

```bash
git clone https://github.com/ayar-12/wizerd.git
cd wizerd
pip install -r requirements.txt
python gesture_launcher.py
```

On first run, it downloads a small hand-tracking model file (`hand_landmarker.task`, a few MB) into your local Application Support folder. Needs internet once; cached afterward.

Press `q` to quit.

## Packaging as a standalone macOS app

A PyInstaller spec file is included:

```bash
pip install pyinstaller
pyinstaller --clean gesture_launcher.spec
```

Produces `dist/GestureLauncher.app` — a double-clickable app with camera permission already declared in its `Info.plist`. Since it's unsigned, macOS will block it the first time; right-click then Open to confirm past that once.

## Tuning gesture sensitivity

All thresholds are constants near the top of `gesture_launcher.py` — raise them if gestures trigger too easily by accident, lower them if deliberate gestures aren't registering:

- `MIN_SPREAD_PX`, `CIRCLE_ROTATION_THRESHOLD`, `CLOSE_LOOP_RATIO` — circle
- `MIN_SWIPE_DIST` — swipe straightness/distance
- Checkmark and zigzag thresholds are inline in their respective functions
- `CLAP_CLOSE_PX`, `CLAP_APART_PX` — how close/far hands need to be for a clap
- `THUMBS_UP_STREAK_NEEDED` — how many consecutive frames the pose must hold

## Known limitations

- Single-hand tracking for the trail gestures (only `hands[0]` is used) — if two hands are visible, the trail follows whichever MediaPipe reports first.
- Gesture thresholds were tuned empirically on one person's webcam/lighting — you'll likely need to adjust them for your own setup.
- No settings UI yet — all tuning is done by editing constants directly.
- Actions are hardcoded to open specific URLs; not yet configurable via a config file.

## Ideas for contributing

- Add more gesture shapes (each is just a new geometry-check function)
- Make actions configurable via a JSON/YAML config instead of hardcoded URLs
- Add a system tray/menu bar mode instead of a persistent camera window
- Support Windows/Linux packaging (currently only the macOS `.app` path is documented)

## License

MIT — see [LICENSE](LICENSE).
