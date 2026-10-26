"""Live webcam fall detection. Shows ONLY the skeleton. Alerts via Telegram.
Setup: export TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=...   (create a bot with @BotFather)
Run:   python live.py        (press q to quit)
"""
import os, time, collections, cv2, numpy as np, torch, requests, mediapipe as mp
from common import *
from train import FallNet

THRESH, CONSEC, COOLDOWN = 0.6, 3, 30
TOKEN, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
mpp, mpd = mp.solutions.pose, mp.solutions.drawing_utils
pose = mpp.Pose(model_complexity=1)
net = FallNet(); net.load_state_dict(torch.load("fallnet.pt")); net.eval()

def alert(img):
    print(">>> FALL ALERT")
    if not (TOKEN and CHAT): return
    _, buf = cv2.imencode(".jpg", img)
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                  data={"chat_id": CHAT, "caption": "Possible fall detected. Please check in."},
                  files={"photo": buf.tobytes()}, timeout=10)

cap = cv2.VideoCapture(0)
buf, hits, last_t, last_alert, prob = collections.deque(maxlen=WIN), 0, 0, 0, 0.0
while True:
    ok, frame = cap.read()
    if not ok: break
    h, w = frame.shape[:2]
    r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    canvas = np.zeros_like(frame)                       # privacy: raw video is never shown or sent
    if r.pose_landmarks:
        mpd.draw_landmarks(canvas, r.pose_landmarks, mpp.POSE_CONNECTIONS)
        if time.time() - last_t >= 1 / FPS:
            last_t = time.time()
            lm = np.array([[p.x, p.y] for p in r.pose_landmarks.landmark])
            buf.append(frame_features(lm, w, h))
            if len(buf) == WIN:
                with torch.no_grad():
                    prob = torch.sigmoid(net(torch.tensor(to_input(np.array(buf)))[None])).item()
                hits = hits + 1 if prob > THRESH else 0
                if hits >= CONSEC and time.time() - last_alert > COOLDOWN:
                    last_alert = time.time(); alert(canvas.copy())
    status = "FALL!" if time.time() - last_alert < 5 else "OK"
    cv2.putText(canvas, f"{status}  p(fall)={prob:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if status != "OK" else (0, 255, 0), 2)
    cv2.imshow("FallGuard (skeleton only)", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"): break