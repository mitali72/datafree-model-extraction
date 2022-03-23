from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from videotransform import *

def get_trainloader(root, num_classes, batch_size, train_split = 0.9, 
	frames_per_clip = 10, step_between_clips=1,
	pin_memory=True):
	# transforms = 
	dataset = Kinetics(
		root=root, frames_per_clip=frames_per_clip,
		step_between_clips=step_between_clips,
		 num_classes=str(num_classes), num_workers=8)
	
	train_set, val_set = random_split(dataset,[int(train_split*len(dataset)), len(dataset)-int((train_split)*len(dataset)) ])
	
	train_loader = DataLoader(train_set, 
		batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,pin_memory=pin_memory)

	val_loader = DataLoader(val_set, 
		batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=pin_memory)
	return train_loader, val_loader

def get_testloader(root, num_classes, batch_size, pin_memory=True):
	dataset = Kinetics(
		root=root, frames_per_clip=10, num_classes=str(num_classes), num_workers=8)
	loader = DataLoader(dataset, 
		batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=pin_memory)
	return loader


# #Normalise mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
# train_pipeline = [
#     dict(type='DecordInit'),
#     dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]
# val_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=32,
#         frame_interval=2,
#         num_clips=1,
#         test_mode=True),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Flip', flip_ratio=0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]