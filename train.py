"""Train the fall classifier. Splits BY VIDEO to avoid leakage."""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from common import *

class FallNet(nn.Module):
    def __init__(self, c=2 * N_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)

if __name__ == "__main__":
    d = np.load("windows.npz")
    X = np.stack([to_input(w) for w in d["X"]]); y = d["y"].astype(np.float32); vid = d["vid"]
    rng = np.random.default_rng(0)
    vids = np.unique(vid); rng.shuffle(vids)
    test_v = set(vids[: max(1, int(0.2 * len(vids)))])
    te = np.array([v in test_v for v in vid])
    Xtr, ytr, Xte, yte = map(torch.tensor, (X[~te], y[~te], X[te], y[te]))
    m = FallNet(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / max(1, (ytr == 1).sum())))
    for ep in range(40):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            xb = Xtr[b] + 0.02 * torch.randn_like(Xtr[b])      # noise augmentation
            opt.zero_grad(); lossf(m(xb), ytr[b]).backward(); opt.step()
        if ep % 10 == 9: print("epoch", ep + 1)
    m.eval()
    with torch.no_grad(): p = (torch.sigmoid(m(Xte)) > 0.5).numpy()
    print(classification_report(yte, p, target_names=["not fall", "fall"]))
    print(confusion_matrix(yte, p))   # screenshot this for Devpost
    torch.save(m.state_dict(), "fallnet.pt")