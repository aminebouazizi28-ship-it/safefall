import os
import threading
import time

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fall_detection import ALERT_P, get_fall_probability

# ---------- SETTINGS ----------
# 0 = camera of this PC.
# To use your friend's camera, put its stream URL instead, for example:
# CAMERA_SOURCE = "http://192.168.1.25:5001/video"
CAMERA_SOURCE = 0
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_lite.task")
# ------------------------------

app = Flask(__name__)

state = {"fall_probability": 0.0, "alert": False, "fall_confirmed": False}
latest_jpeg = None
lock = threading.Lock()


def draw_overlay(frame, lm, r):
    h, w = frame.shape[:2]
    if lm:
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
    if r["fall_confirmed"]:
        cv2.putText(frame, "FALL DETECTED", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif r["probability"] >= ALERT_P:
        cv2.putText(frame, "HIGH FALL RISK", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)


def camera_loop():
    global latest_jpeg

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    start = time.time()
    last_ts = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            # camera lost: wait and try to reconnect
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int((time.time() - start) * 1000)
        if ts <= last_ts:          # MediaPipe needs strictly increasing timestamps
            ts = last_ts + 1
        last_ts = ts

        res = landmarker.detect_for_video(img, ts)
        lm = res.pose_landmarks[0] if res.pose_landmarks else None
        r = get_fall_probability(lm, w, h)

        draw_overlay(frame, lm, r)
        ok, buf = cv2.imencode(".jpg", frame)

        with lock:
            state["fall_probability"] = r["probability"]
            state["alert"] = r["probability"] >= ALERT_P
            state["fall_confirmed"] = bool(r["fall_confirmed"])
            if ok:
                latest_jpeg = buf.tobytes()


def mjpeg():
    while True:
        with lock:
            jpeg = latest_jpeg
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def status():
    with lock:
        return jsonify(state)


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    # debug=False is important: debug mode would start the camera twice
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
