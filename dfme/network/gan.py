from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
use_gpu = True if torch.cuda.is_available() else False

import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, activation=None, final_bn=True):
        super(GeneratorA, self).__init__() 

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False) 
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False) 
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x :
            return img
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            return self.activation(img)

class GeneratorC(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(GeneratorC, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3,1,1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class GeneratorImageOurs(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, img_size=128, activation=None):
        super().__init__()
        self.img_size = img_size
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='DTD',
                       pretrained=True, useGPU=use_gpu)
        self.modelG = self.model.getNetG()
        self.modelG.addScale(128)
        self.modelG.setNewAlpha(0.0)
        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation
        


    def forward(self,z, pre_x = None):
        # Generate images batch*C*H*W
        output = self.modelG(z)[:,:,:32,:32]
        output = output.unsqueeze(1)
        
        if pre_x :
            return output
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            return self.activation(output)

        
class VideoGenerator(nn.Module): #input intitialization: model = VideoGenerator(3,a,b,c,video_length); a+b+c = 559
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64, activation=None,device=None):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.activation = activation
        dim_z = dim_z_motion + dim_z_category + dim_z_content
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='DTD',
                       pretrained=True, useGPU=False)
        self.modelG = self.model.getNetG()
        self.modelG.addScale(128)
        self.modelG.setNewAlpha(0.0)

#         self.upsample = nn.Sequential(
#             nn.ConvTranspose2d(self.n_channels, self.n_channels, 4,2,3),
#             nn.BatchNorm2d(self.n_channels),
#             nn.AvgPool2d(kernel_size = (16,16), stride = (2,2)),
#             nn.ReLU(True),
#             nn.Conv2d(self.n_channels, self.n_channels, 8, 1),
#             nn.BatchNorm2d(self.n_channels),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(self.n_channels, self.n_channels, 4,2,1),
#             nn.ReLU(True)
#         )
        
                                     

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t.to(self.device), h_t[-1].to(self.device)))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)
        
        return z_m

    def sample_z_categ(self, num_samples, video_len):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)
        # num samples (int), num_samples.shape=[1,559]
        n_samples = num_samples#.detach().cpu().numpy().shape[0]
        # b = num_samples.detach().cpu().numpy().shape[1]

        classes_to_generate = np.random.randint(self.dim_z_category, size=(n_samples))
        one_hot = np.zeros((n_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(n_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        
        # num samples (int), num_samples.shape=[1,559]
        n_samples = num_samples#.detach().cpu().numpy().shape[0]
        content = np.random.normal(0, 1, (n_samples, self.dim_z_content)).astype(np.float32)
        
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples, video_len=None):
        z_content = self.sample_z_content(num_samples, video_len).to(self.device)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_category = z_category.to(self.device)
        z_motion = self.sample_z_m(num_samples, video_len).to(self.device)
        
        
        if z_category is not None:
            
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z = torch.cat([z_content, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, num_samples, pre_x = None, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(num_samples, video_len)
        
        
        h = self.modelG(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view(h.size(0) // video_len, video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()
        
        # h = h.permute(0, 2, 1, 3, 4)
        
        if pre_x :
            return h
        else:
            return self.activation(h)
        # return h#, Variable(z_category_labels, requires_grad=False)

    def sample_images(self, num_samples):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None

    def get_gru_initial_state(self, num_samples):
        # num samples (int), num_samples.shape=[1,559]
        n_samples = num_samples#.detach().cpu().numpy().shape[0]
        return Variable(T.FloatTensor(n_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples):
        # num samples (int), num_samples.shape=[1,559]
        n_samples = num_samples#num_samples.detach().cpu().numpy().shape[1]
        return Variable(T.FloatTensor(n_samples, self.dim_z_motion).normal_())
