import torch
from configs.submission import get_cfg
from core.FlowFormer import build_flowformer
from utils.utils import InputPadder
import time

print("build FlowFormer...")
start = time.time()
cfg = get_cfg()
model = build_flowformer(cfg)
model_state = torch.load(cfg.model)
model_state = {l.replace("module.", ''): r for l, r in model_state.items()}
model.load_state_dict(model_state)
model.cuda()
model.eval()
print("{:.3f} s".format(time.time() - start))

def compute_flow(image1, image2, weights=None):
	with torch.no_grad():
		if weights is None:     # no tile
			padder = InputPadder(image1.shape)
			image1, image2 = padder.pad(image1, image2)

			flow_pre, _ = model(image1, image2)

			flow_pre = padder.unpad(flow_pre)

	return flow_pre

