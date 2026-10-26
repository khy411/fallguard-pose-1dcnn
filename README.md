# fallguard-pose-1dcnn

Privacy-first fall detection. A camera feed is turned into a pose skeleton on the device, a small 1D-CNN decides whether the last two seconds of movement look like a fall, and an alert fires. No video is shown, stored, or sent.

**Live demo (runs fully in your browser):** https://khy411.github.io/fallguard-pose-1dcnn/

## The problem

Falls are a leading cause of injury for older adults, and the worst outcomes often come from lying on the floor for a long time before anyone notices. Wearables get forgotten or left uncharged, and ordinary cameras are a hard sell in a bedroom or bathroom. FallGuard keeps the camera but throws away the picture: only skeleton keypoints are ever used.

Who it could help: older adults living alone, their families, and care homes that want monitoring without recording residents.

## How it works

1. **Pose estimation (pretrained):** MediaPipe Pose Landmarker returns 33 body landmarks per frame.
2. **Features (mine):** 13 joints are centered on the hips and scaled by torso length, plus hip position, torso angle, and body aspect ratio. Velocity channels are added. Frames are sampled at 15 fps into 2-second windows (30 frames).
3. **Classifier (mine):** a small 1D-CNN over each window outputs a fall probability.
4. **Alert logic:** an alarm fires after 3 consecutive windows above a threshold of 0.6, with a cooldown so one fall raises one alarm.
5. **Two runtimes, one set of features:**
   - Python (`live.py`): OpenCV webcam with a skeleton-only view. When a fall is detected it sends a Telegram alert to a caregiver's phone with a skeleton-only snapshot, so no camera image is ever sent. A cooldown prevents repeat alerts for the same fall.
   - Browser (`docs/`): MediaPipe Tasks for the web plus ONNX Runtime Web, so the model runs entirely on the visitor's device. It shows an on-screen banner and beep when a fall is detected. `docs/features.js` is a line-by-line port of `common.py`, and I checked that both produce identical numbers.

## Results

Evaluated on held-out videos (the train/test split is by video, so no clip appears on both sides). Window-level results on 554 test windows:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not fall | 0.95 | 0.98 | 0.97 |
| Fall | 0.91 | 0.74 | 0.81 |

Accuracy 0.94. Confusion matrix (rows are true class, columns are predicted): `[[452, 7], [25, 70]]`.

The test set is small, so treat these numbers as indicative. They are window-level results: one fall produces many windows and the alarm only needs three consecutive hits, so the per-fall catch rate is likely higher than the 0.74 window recall, but I have not measured it formally.

## Known limitations

- One person in frame. Multiple people are not handled.
- Trained on a small public dataset plus my own recordings, mostly from a laptop-webcam viewpoint. Other camera angles and heights will perform worse.
- Deliberately lying down quickly, or falls out of frame, can cause false alarms or misses.
- The web version uses MediaPipe's web pose model, which may jitter slightly differently from the Python one. The threshold of 0.6 was tuned on the Python version.
- Telegram alerts are only available in the Python app. The website shows an on-screen banner and beep instead, because a browser page cannot safely hold a bot token.
- This is a prototype, not a medical device.

## Planned (not built yet)

- An "I'm OK" button to cancel a false alarm before anyone is notified.
- Multi-person support.
- Detecting a long lie on the floor after a fall.
- A phone app.

## Run it yourself

Python 3.12 is recommended (MediaPipe 0.10.14 does not install on 3.13).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data:** videos are not included in this repo. Put clips in `data/fall/` (one fall per clip) and `data/adl/` (daily activities), then:

```bash
python extract.py     # keypoint windows -> windows.npz
python train.py       # trains, prints the report, saves fallnet.pt
python live.py        # webcam demo, press q to quit
```

### Telegram alerts (Python app only)

1. In Telegram, message @BotFather, send `/newbot`, and copy the bot token.
2. Open your new bot and press Start, then send it any message.
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and copy your `chat.id`.
4. Set both values in the terminal before running the app:

```powershell
$env:TELEGRAM_TOKEN="your-bot-token"
$env:TELEGRAM_CHAT_ID="your-chat-id"
python live.py
```

On a detected fall the bot sends the message "Possible fall detected. Please check in." along with the skeleton snapshot. If the two variables are not set, the app still runs and shows the alarm on screen. The token is read from environment variables and is never stored in the repo.

### Rebuild the website model

```bash
pip install onnx onnxruntime
python export_onnx.py          # writes docs/fallnet.onnx
cd docs && python -m http.server 8000
```

Then open http://localhost:8000. The site is served from the `/docs` folder with GitHub Pages.

If MediaPipe fails with a protobuf error after installing `onnx`, run `pip install "protobuf>=4.25.3,<5"`.

## Project layout

```
common.py        pose landmarks -> features
extract.py       videos -> labeled windows
train.py         1D-CNN training and evaluation
live.py          Python webcam app with Telegram alerts
export_onnx.py   PyTorch -> ONNX for the website
docs/            static website (index.html, features.js, fallnet.onnx)
```

## Credits

- **Pose estimation:** Google MediaPipe (Pose Landmarker), pretrained.
- **Training data:** GMDCSA-24 fall detection dataset (https://pmc.ncbi.nlm.nih.gov/articles/PMC11416611), plus my own recordings.
- **Libraries:** PyTorch, OpenCV, NumPy, scikit-learn, ONNX Runtime (Python and Web), Telegram Bot API.
- **Fonts:** Bricolage Grotesque and Public Sans (Google Fonts).
- **My contribution:** the feature design, fall-moment labeling, classifier training and evaluation, alert logic, the Python app, and the in-browser port.
- **AI assistance:** Claude (Anthropic) helped with debugging. I recorded the data, trained and tested the model, and verified the results.