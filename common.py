"""Shared feature code: pose landmarks -> per-frame features -> model input."""
import numpy as np

FPS = 15 # we sample every video/webcam at ~15 fps
WIN = 30 # 2-second window
IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] # nose, shoulders, elbows, wrists, hips, knees, ankles
N_FEAT = len(IDX) * 2 + 5 # coords + hip_x, hip_y, sin, cos, aspect


def frame_features(lm, w, h):
    """lm: (33,2+) array of normalized x,y. Returns (N_FEAT,) vector."""
    px = np.stack([lm[:, 0] * w, lm[:, 1] * h], 1)
    hip = (px[23] + px[24]) / 2
    sh = (px[11] + px[12]) / 2
    v = sh - hip
    torso = max(np.linalg.norm(v), 1e-3)
    pts = (px[IDX] - hip) / torso                     # centered, scale-invariant
    sin, cos = v[0] / torso, -v[1] / torso            # torso angle from vertical (up = +)
    bw = np.ptp(px[IDX, 0]); bh = max(np.ptp(px[IDX, 1]), 1e-3)
    aspect = np.clip(bw / bh, 0, 5)
    return np.concatenate([pts.ravel(), [hip[0] / w, hip[1] / h, sin, cos, aspect]]).astype(np.float32)


def to_input(win):
    """win: (WIN, N_FEAT) -> (2*N_FEAT, WIN) with velocity channels."""
    d = np.diff(win, axis=0, prepend=win[:1])
    return np.concatenate([win, d], 1).T.astype(np.float32)