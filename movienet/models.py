import torch
from stam_pytorch import STAM
import pytorchvideo.models.resnet as videoresnet ##Only works with python 3.7
from movinetapi import *
import torch.nn as nn
# from swintapi import *
from swin_transformer_api import SwinT_Kinetics

def get_student(image_size, classifier="stam", num_classes=400): 
	if "x3d" in classifier:
		model = torch.hub.load('facebookresearch/pytorchvideo', classifier, pretrained=False)
		return model
	elif "resnet" in classifier: ##Only works with python 3.7
		try:
			depth = int(classifier[6:])
		except Exception as e:
			raise "Please enter correct student model"
		model = videoresnet.create_resnet( 
			  input_channel=3, 
			  model_depth=depth, 
			  model_num_class=num_classes, 
			  norm=nn.BatchNorm3d,
			  activation=nn.ReLU,
		  )#source: https://pytorchvideo.readthedocs.io/en/latest/api/models/resnet.html#pytorchvideo.models.resnet.create_resnet
		# if pretrained:
		# 	#TODO: load pretrained on imagenet
		# 	#source: https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/checkpoint.py#L142
		# 	pass
		return model

	elif classifier == "stam":
		model = STAM(
		dim = 512,
		image_size = image_size,     # size of image
		patch_size = 32,      # patch size
		num_frames = 10,       # number of image frames, selected out of video
		space_depth = 12,     # depth of vision transformer
		space_heads = 8,      # heads of vision transformer
		space_mlp_dim = 2048, # feedforward hidden dimension of vision transformer
		time_depth = 6,       # depth of time transformer (in paper, it was shallower, 6)
		time_heads = 8,       # heads of time transformer
		time_mlp_dim = 2048,  # feedforward hidden dimension of time transformer
		num_classes = num_classes,    # number of output classes
		space_dim_head = 64,  # space transformer head dimension
		time_dim_head = 64,   # time transformer head dimension
		dropout = 0.4,         # dropout
		emb_dropout = 0.2      # embedding dropout
		)
		return model
	else:
		raise "Enter valid model name"


def get_teacher(num_classes, device):
	if(num_classes==600):
		return Movinet(device)

	elif(num_classes==400):
		model = SwinT_Kinetics()
		model.load_state_dict(torch.load('swint_final_weights.pt')) 
		model.eval().to(device)
		return model