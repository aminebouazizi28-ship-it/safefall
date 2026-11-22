import cv2, time, math
from collections import deque
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- PARAMETERS (tune these in Hour 4) ----------
WINDOW_S = 4.0      # seconds of history used to judge instability
MIN_SPAN = 1.5      # seconds of data needed before a risk is computed

# Value at which each instability signal counts as "maximum risk"
SWAY_MAX = 12.0     # std of the trunk angle over the window (degrees)
LEAN_MAX = 40.0     # fast trunk tilt, 90th percentile (degrees/second)
WOBBLE_MAX = 0.08   # side-to-side hip wobble, in torso lengths
SPEED_MAX = 0.8     # head descent speed (frame-heights/second)

# Weights of each signal in the risk score (sum = 1)
W_SWAY, W_LEAN, W_WOBBLE, W_SPEED = 0.30, 0.25, 0.20, 0.25

ALERT_P = 0.6       # risk above this = warn that a fall is likely

# Last stage: a fall that really happened pushes the risk to ~1
ANGLE_THR = 60      # trunk angle above this = lying down
HEAD_LOW = 0.35     # head very low in the frame
SPEED_THR = 0.8     # fast drop
DROP_MEMORY = 4.0   # seconds a fast drop is remembered
MOVE_THR = 0.05     # movement below this = motionless
STILL_TIME = 3.0    # seconds of stillness to confirm
# -------------------------------------------------------

KEY_POINTS = [0, 11, 12, 23, 24]  # nose, shoulders, hips


def clip01(x):
    return max(0.0, min(1.0, x))


class FallDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_head = None
        self.prev_pts = None
        self.prev_t = None
        self.speed = 0.0
        self.last_drop_t = -1e9
        self.still_since = None
        self.confirmed = False
        self.window = deque()      # (time, trunk angle, hip x / torso length)
        self.risk = 0.0

    def update(self, lm, w, h, now=None):
        """lm = landmarks of ONE person (or None if nobody is seen)."""
        now = time.time() if now is None else now
        if lm is None:
            self.reset()
            return {"probability": 0.0, "fall_confirmed": False, "angle": 0.0,
                    "head": 0.0, "speed": 0.0, "immobile_s": 0.0,
                    "sway": 0.0, "lean_rate": 0.0, "hip_wobble": 0.0}

        # --- basic values ---
        sx = (lm[11].x + lm[12].x) / 2 * w
        sy = (lm[11].y + lm[12].y) / 2 * h
        hx = (lm[23].x + lm[24].x) / 2 * w
        hy = (lm[23].y + lm[24].y) / 2 * h
        torso = max(math.hypot(sx - hx, sy - hy), 1.0)
        angle = math.degrees(math.atan2(abs(sx - hx), hy - sy))
        head = 1 - lm[0].y

        pts = [(lm[i].x, lm[i].y) for i in KEY_POINTS]
        movement = 0.0
        if self.prev_t is not None and now > self.prev_t:
            dt = now - self.prev_t
            raw = (self.prev_head - head) / dt
            self.speed = 0.7 * self.speed + 0.3 * raw
            movement = sum(math.hypot(a[0] - b[0], a[1] - b[1])
                           for a, b in zip(pts, self.prev_pts)) / len(pts) / dt
        self.prev_head, self.prev_pts, self.prev_t = head, pts, now

        # --- sliding window of the last few seconds ---
        self.window.append((now, angle, hx / torso))
        while self.window and now - self.window[0][0] > WINDOW_S:
            self.window.popleft()

        sway = lean = wobble = 0.0
        if len(self.window) >= 10 and now - self.window[0][0] >= MIN_SPAN:
            ts = np.array([p[0] for p in self.window])
            angs = np.array([p[1] for p in self.window])
            hxs = np.array([p[2] for p in self.window])

            sway = float(np.std(angs))                              # trunk wobble
            dts = np.maximum(np.diff(ts), 1e-3)
            lean = float(np.percentile(np.abs(np.diff(angs)) / dts, 90))  # sudden tilts
            trend = np.polyval(np.polyfit(ts - ts[0], hxs, 1), ts - ts[0])
            wobble = float(np.std(hxs - trend))                     # hips swaying sideways

            score = (W_SWAY * clip01(sway / SWAY_MAX)
                     + W_LEAN * clip01(lean / LEAN_MAX)
                     + W_WOBBLE * clip01(wobble / WOBBLE_MAX)
                     + W_SPEED * clip01(max(self.speed, 0.0) / SPEED_MAX))
            self.risk = 0.85 * self.risk + 0.15 * score             # smooth, no flicker
        else:
            self.risk = 0.0                                         # not enough data yet

        # --- last stage: confirm a fall that actually happened ---
        if self.speed > SPEED_THR:
            self.last_drop_t = now
        recent_drop = (now - self.last_drop_t) < DROP_MEMORY
        lying = angle > ANGLE_THR or head < HEAD_LOW

        if lying and movement < MOVE_THR:
            if self.still_since is None:
                self.still_since = now
        else:
            self.still_since = None
        immobile_s = 0.0 if self.still_since is None else now - self.still_since

        if recent_drop and lying and immobile_s >= STILL_TIME:
            self.confirmed = True
        if not lying:
            self.confirmed = False

        p = self.risk
        if self.confirmed:
            p = max(p, 0.95)

        return {"probability": round(clip01(p), 2), "fall_confirmed": self.confirmed,
                "angle": angle, "head": head, "speed": self.speed,
                "immobile_s": immobile_s, "sway": sway, "lean_rate": lean,
                "hip_wobble": wobble}


# ---------- Function to give to Person 2 (same name and keys as before) ----------
_detector = FallDetector()

def get_fall_probability(landmarks, frame_width, frame_height):
    """Call once per frame with the landmarks of one person (or None).
    Returns a dict: probability (0..1 fall risk), fall_confirmed (bool), plus debug values."""
    return _detector.update(landmarks, frame_width, frame_height)


# ---------- Demo with webcam ----------
if __name__ == "__main__":
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        running_mode=vision.RunningMode.VIDEO)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(img, int((time.time() - start) * 1000))

        lm = res.pose_landmarks[0] if res.pose_landmarks else None
        r = get_fall_probability(lm, w, h)

        if lm:
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
            lines = [f"Sway: {r['sway']:.1f}", f"Lean rate: {r['lean_rate']:.0f}",
                     f"Hip wobble: {r['hip_wobble']:.3f}", f"Speed: {r['speed']:.2f}",
                     f"FALL RISK: {r['probability']:.2f}"]
            for i, t in enumerate(lines):
                cv2.putText(frame, t, (10, 30 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # risk bar
            bar = int(200 * r["probability"])
            color = (0, 0, 255) if r["probability"] >= ALERT_P else (0, 200, 0)
            cv2.rectangle(frame, (10, 190), (210, 210), (255, 255, 255), 1)
            cv2.rectangle(frame, (10, 190), (10 + bar, 210), color, -1)

        if r["fall_confirmed"]:
            cv2.putText(frame, "FALL DETECTED", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif r["probability"] >= ALERT_P:
            cv2.putText(frame, "HIGH FALL RISK", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 255), 3)

        cv2.imshow("pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
