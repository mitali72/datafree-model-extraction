from movinets import MoViNet
from movinets.config import _C
import torch

class Movinet():
	def __init__(self, device):
		self.model = MoViNet(_C.MODEL.MoViNetA2, causal=False, pretrained=True).to(device).eval()

	def shift(self, pts):
		pts_min = torch.reshape(torch.amin(pts, dim=(1, 2, 3)), (pts.shape[0], 1, 1, 1, pts.shape[4]))
		pts_max = torch.reshape(torch.amax(pts, axis=(1, 2, 3)), (pts.shape[0], 1, 1, 1, pts.shape[4]))
		pts_norm = (pts + pts_min) / (pts_max - pts_min)
		#pts = (pts+ pts_min)/(pts_max-pts_min)
		return pts_norm

	def __call__(self, x):
		return self.model(self.shift(x))

