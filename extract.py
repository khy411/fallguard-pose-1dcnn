"""Turn videos into labeled keypoint windows.
Layout:  data/fall/*.mp4   (clips containing a fall)
         data/adl/*.mp4    (daily activities: walk, sit, bend, lie down, pick up...)
Run:     python extract.py
"""
import glob, os, cv2, numpy as np, mediapipe as mp
from common import *

pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)


def video_feats(path):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src / FPS))
    feats, last, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if i % step == 0:
            h, w = frame.shape[:2]
            r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if r.pose_landmarks:
                lm = np.array([[p.x, p.y] for p in r.pose_landmarks.landmark])
                last = frame_features(lm, w, h)
            if last is not None:
                feats.append(last)
        i += 1
    return np.array(feats)


def fall_peak(f):
    """Frame of fastest torso tilt + hip drop (the moment of the fall)."""
    hip_y, sin = f[:, -4], f[:, -3]
    k = 5
    score = np.abs(hip_y[k:] - hip_y[:-k]) + np.abs(sin[k:] - sin[:-k])
    return int(np.argmax(score)) + k // 2


X, y, vid = [], [], []
for label, folder in [(1, "fall"), (0, "adl")]:
    files = sorted(sum([glob.glob(f"data/{folder}/*.{e}") for e in ("mp4", "avi", "mov")], []))
    for path in files:
        f = video_feats(path)
        if len(f) < WIN: continue
        peak = fall_peak(f) if label else None
        for s in range(0, len(f) - WIN + 1, 3):
            if label and not (s + 6 <= peak < s + WIN - 6):
                continue   # in fall clips keep only windows containing the fall itself
            X.append(f[s:s + WIN]); y.append(label); vid.append(path)
        print(path, len(f), "frames")
np.savez("windows.npz", X=np.array(X), y=np.array(y), vid=np.array(vid))
print("windows:", len(X), "falls:", int(sum(y)))