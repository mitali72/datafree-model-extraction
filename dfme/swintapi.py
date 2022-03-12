from mmaction.core import OutputHook
from mmcv.parallel import scatter
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer 

class SwinT():
	def __init__(self, device):
		super().__init__()
		config_file = 'checkpoints/swinT/swin_tiny_patch244_window877_kinetics400_1k.py' #for swin-T
		checkpoint_file = 'checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
		self.device = torch.device(device)
		model = init_recognizer(config_file, checkpoint_file, device=device)
		model.eval()

	def SwinTApi(self, data, as_tensor= True):
	  '''data is a tensor: nxtxcxhxw'''
	  data = data.permute(0,2,1,3,4)
	  if next(model.parameters()).is_cuda:
	      # scatter to specified GPU
	      data = scatter(data, [device])[0]

	  # forward the model
	  with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
	      with torch.no_grad():
	          scores = model(data,return_loss=False)[0]
	      return h.layer_outputs if outputs else None

	def __call__(self, x):
		ans = self.SwinTApi(x)
		return ans