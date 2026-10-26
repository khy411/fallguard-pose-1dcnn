import json, numpy as np, torch
from common import *
from train import FallNet

m = FallNet(); m.load_state_dict(torch.load("fallnet.pt")); m.eval()
dummy = torch.zeros(1, 2 * N_FEAT, WIN)
try:
    torch.onnx.export(m, dummy, "docs/fallnet.onnx", input_names=["x"], output_names=["logit"], opset_version=17, dynamo=False)
except TypeError:
    torch.onnx.export(m, dummy, "docs/fallnet.onnx", input_names=["x"], output_names=["logit"], opset_version=17)
try:
    import onnxruntime as ort
    x = np.random.randn(1, 2 * N_FEAT, WIN).astype(np.float32)
    a = ort.InferenceSession("docs/fallnet.onnx").run(None, {"x": x})[0]
    b = m(torch.tensor(x)).detach().numpy()
    print("onnx vs torch max diff:", float(np.abs(a - b).max()))
except ImportError:
    pass
print("wrote docs/fallnet.onnx")