import cv2, time, csv, statistics
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fall_detection import FallDetector, ALERT_P

# Keys:  S = start a STEADY trial (walk normally, sit, stand, bend)
#        U = start an UNSTABLE trial (sway, stagger, stumble, controlled fall on mattress)
#        SPACE = end and save the trial      ESC = quit and print suggestions
WARMUP = 2.0   # seconds after pressing S/U before recording starts (the risk needs history)

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

detector = FallDetector()
trial, trials, msg = None, [], "Press S (steady) or U (unstable) to start a trial"

cap = cv2.VideoCapture(0)
start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(img, int((now - start) * 1000))
    lm = res.pose_landmarks[0] if res.pose_landmarks else None
    r = detector.update(lm, w, h)

    if lm:
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

    if trial is None:
        status, color = "Idle", (255, 255, 0)
    elif now - trial["t0"] < WARMUP:
        status, color = "WAIT... stand still", (0, 200, 255)
    else:
        trial["sway"] = max(trial["sway"], r["sway"])
        trial["lean"] = max(trial["lean"], r["lean_rate"])
        trial["wobble"] = max(trial["wobble"], r["hip_wobble"])
        trial["speed"] = max(trial["speed"], r["speed"])
        trial["prob"] = max(trial["prob"], r["probability"])
        status, color = f"GO - recording {trial['label'].upper()} (SPACE to end)", (0, 0, 255)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"risk {r['probability']:.2f}  sway {r['sway']:.1f}  lean {r['lean_rate']:.0f}  wobble {r['hip_wobble']:.3f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imshow("tuning", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key in (ord("s"), ord("u")) and trial is None:
        detector.reset()
        trial = {"label": "steady" if key == ord("s") else "unstable", "t0": now,
                 "sway": 0.0, "lean": 0.0, "wobble": 0.0, "speed": 0.0, "prob": 0.0}
    elif key == 32 and trial:
        trial["alerted"] = trial["prob"] >= ALERT_P
        trial.pop("t0")
        trials.append(trial)
        msg = f"Saved {trial['label']} #{len(trials)}: peak risk {trial['prob']:.2f}"
        trial = None

cap.release()
cv2.destroyAllWindows()

# ---------- summary ----------
steady = [t for t in trials if t["label"] == "steady"]
unstable = [t for t in trials if t["label"] == "unstable"]
if trials:
    with open("trials.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        wr.writeheader()
        wr.writerows(trials)

print(f"\nUnstable trials flagged: {sum(t['alerted'] for t in unstable)}/{len(unstable)}")
print(f"False alarms (steady):   {sum(t['alerted'] for t in steady)}/{len(steady)}")

if steady and unstable:
    print("\nSuggested values for fall_detection.py (median peak of your unstable trials):")
    for name, key in [("SWAY_MAX", "sway"), ("LEAN_MAX", "lean"),
                      ("WOBBLE_MAX", "wobble"), ("SPEED_MAX", "speed")]:
        s_max = max(t[key] for t in steady)
        u_med = statistics.median(t[key] for t in unstable)
        flag = "" if u_med > s_max else "   <- weak: steady peaks as high as unstable"
        print(f"  {name} = {u_med:.2f}    (steady peak max was {s_max:.2f}){flag}")

    s_p = max(t["prob"] for t in steady)
    u_p = min(t["prob"] for t in unstable)
    if u_p > s_p:
        print(f"\nALERT_P = {(s_p + u_p) / 2:.2f}   (steady peaks <= {s_p:.2f}, weakest unstable {u_p:.2f})")
    else:
        print(f"\nALERT_P: OVERLAP (steady up to {s_p:.2f}, weakest unstable {u_p:.2f}). "
              "Apply the *_MAX values above, then run again.")
print("Details saved in trials.csv")
