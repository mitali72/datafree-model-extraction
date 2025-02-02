from __future__ import print_function
import argparse, ipdb, json

from parso import parse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import network
from dataloader import get_dataloader
import os, random
import numpy as np
import torchvision
from pprint import pprint
from time import time
import copy

from approximate_gradients_copy import *

import torchvision.models as models
from my_utils import *
import pdb

print("torch version", torch.__version__)

def myprint(a):
    """Log the print statements"""
    global file
    print(a); file.write(a); file.write("\n"); file.flush()


def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=-1)
        t_logit = F.softmax(t_logit, dim=-1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def generator_loss(args, s_logit, t_logit,  z = None, z_logit = None, reduction="mean"):
    assert 0 
    
    loss = - F.l1_loss( s_logit, t_logit , reduction=reduction) 
    
            
    return loss


def train(args, teacher, student, student2, generator, device, optimizer, epoch, best_loss_prev, number_epochs, info_loss_coef = 0):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()
    student2.train()

    optimizer_S, optimizer_S2,  optimizer_G, optimizer_RNN = optimizer
    #decayRate = 0.96
    #optimGScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G, gamma=decayRate)
    gradients = []
    # kl_loss = nn.KLDivLoss(reduction="batchmean")
    best_loss = best_loss_prev
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            #Sample Random Noise
            z = args.batch_size#torch.randn((args.batch_size, args.nz)).to(device)
            #print(z.shape)
            #optimizer_G.zero_grad()
            optimizer_RNN.zero_grad()
            generator.train()
            #Get fake image from generator
            # print("Shape of z:\n", z.shape)
            fake, fake_labels, fake_labels_ohe = generator.sample_videos(z, pre_x=args.approx_grad)
            fake.to(device) # pre_x returns the output of G before applying the activation
            #fake_2, xyz =  generator.sample_images(z)
            #fake_2 = fake_2.repeat(1, 10, 1, 1, 1) #add repeat to image to get video
            #fake_2.to(device) # pre_x returns the output of G before applying the activation
            #print(z.shape, fake.shape)
            #pdb.set_trace()
            #print(fake_labels.size())
            #print(fake_labels_ohe.size())
            ## APPROX GRADIENT
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student, fake, fake_labels_ohe, epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, device=device, pre_x=True, lambd = info_loss_coef)
            fake.backward(approx_grad_wrt_x.to(device))
            optimizer_RNN.step()

            #approx_grad_wrt_x_2, loss_G_2 = estimate_gradient_objective(args, teacher, student2, fake_2, fake_labels_ohe, epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, device=device, pre_x=True, lambd = 0)
            #fake_2.backward(approx_grad_wrt_x_2.to(device))
            #optimizer_G.step()

            if i == 0 and args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(args, fake)

        for _ in range(args.g_iter):
            #Sample Random Noise
            z = args.batch_size#torch.randn((args.batch_size, args.nz)).to(device)
            #print(z.shape)
            optimizer_G.zero_grad()
            #optimizer_RNN.zero_grad()
            generator.train()
            #Get fake image from generator
            # print("Shape of z:\n", z.shape)
            #fake, fake_labels, fake_labels_ohe = generator.sample_videos(z, pre_x=args.approx_grad)
            #fake.to(device) # pre_x returns the output of G before applying the activation
            fake_2, xyz =  generator.sample_images(z)
            fake_2 = fake_2.repeat(1, 10, 1, 1, 1) #add repeat to image to get video
            fake_2.to(device) # pre_x returns the output of G before applying the activation
            #print(z.shape, fake.shape)
            #pdb.set_trace()
            #print(fake_labels.size())
            #print(fake_labels_ohe.size())
            ## APPROX GRADIENT
            #approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student, fake, fake_labels_ohe, epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, device=device, pre_x=True, lambd = info_loss_coef)
            #fake.backward(approx_grad_wrt_x.to(device))
            #optimizer_RNN.step()

            approx_grad_wrt_x_2, loss_G_2 = estimate_gradient_objective(args, teacher, student2, fake_2, fake_labels_ohe, epsilon = args.grad_epsilon, m = args.grad_m, num_classes=args.num_classes, device=device, pre_x=True, lambd = 0)
            fake_2.backward(approx_grad_wrt_x_2.to(device))
            optimizer_G.step()

            if i == 0 and args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(args, fake)

        for _, _ in enumerate(range(args.d_iter)):
            z = args.batch_size#torch.randn((args.batch_size, args.nz)).to(device)
            #if i%2==0:
            fake, xyz, abcc = generator.sample_videos(z, pre_x=args.approx_grad)
            #else:
            fake_img, xyz  = generator.sample_images(z)
            fake.detach()
            fake_img.detach()
            optimizer_S.zero_grad()
            optimizer_S2.zero_grad()

            # with torch.no_grad(): 
            if args.num_classes==600:
                
                '''fake => generator output'''
                fake_min = torch.reshape(torch.amin(fake, dim=(1, 2, 3)), (fake.shape[0], 1, 1, 1, fake.shape[4]))
                fake_max = torch.reshape(torch.amax(fake, axis=(1, 2, 3)), (fake.shape[0], 1, 1, 1, fake.shape[4]))
                fake_norm = (fake + fake_min) / (fake_max - fake_min)
                
                teacher.clean_activation_buffers()
                t_logit = teacher(fake_norm.permute(0,2,1,3,4)).to(device)
                
                fake_min_img = torch.reshape(torch.amin(fake_img, dim=(1, 2, 3)), (fake_img.shape[0], 1, 1, 1, fake_img.shape[4]))
                fake_max_img = torch.reshape(torch.amax(fake_img, axis=(1, 2, 3)), (fake_img.shape[0], 1, 1, 1, fake_img.shape[4]))
                fake_norm_img = (fake_img + fake_min_img) / (fake_max_img - fake_min_img)
                
                teacher.clean_activation_buffers()
                t_logit_img = teacher(fake_norm_img.permute(0,2,1,3,4)).to(device)
                

                ''' # prev tensorflow
                try:
                    from functions import tf_to_torch, torch_to_tf                  
                except (ImportError, ModuleNotFoundError):
                    raise "ERROR!!!"
                fake_tf = torch_to_tf(fake)

                fake_tfmin = tf.reshape(tf.reduce_min(fake_tf,axis = [1,2,3]),[fake_tf.shape[0],1,1,1,fake_tf.shape[4]])
                fake_tfmax = tf.reshape(tf.reduce_max(fake_tf,axis = [1,2,3]),[fake_tf.shape[0],1,1,1,fake_tf.shape[4]])
                fake_tf = (fake_tf + fake_tfmin)/(fake_tfmax-fake_tfmin)

                tf_logit = teacher(fake_tf)
                # print("*"*10, tf_logit.shape, "*"*10);
                
                # tf tensor 'tf_logit' to pytorch tensor 't_logit'
                # t_logit = torch.tensor(tf_logit.numpy())
                # t_logit = t_logit.to(device)

                t_logit = tf_to_torch(tf_logit)
                t_logit = t_logit.to(device)
                #should we take loss between softmax of t_logit and z?
                # teacher_loss = kl_loss(t_logit,z)
                '''
                
            else:
                t_logit = teacher(fake.permute(0,2,1,3,4)).to(device)
                t_logit_img = teacher(fake_img.permute(0,2,1,3,4)).to(device)
                # t_logit = teacher(fake)


            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                t_logit_img = F.log_softmax(t_logit_img, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                    t_logit_img -= t_logit_img.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()
                    t_logit_img -= t_logit_img.mean(dim=1).view(-1, 1).detach()

            # fake= fake.repeat(1, 10, 1, 1, 1) #add repeat to image to get video
        
            # print(torch.argmax(s_logit, 1))
            fake_img = fake_img.repeat((1,10,1,1,1))
            s_logit = student(fake)#.permute(0,2,1,3,4))
            s_logit_img = student2(fake_img)#.permute(0,2,1,3,4))
            
            loss_S = student_loss(args, s_logit, t_logit)#+teacher_loss
            loss_S_img = student_loss(args, s_logit_img, t_logit_img)#+teacher_loss

            loss_S.backward()
            loss_S_img.backward()

            # print('before S2 opt')
            optimizer_S2.step()
            # print('after S2 opt, before S opt')
            optimizer_S.step()
            # print('after S opt')

            if loss_S.item() < best_loss:
                best_loss = loss_S.item()
                name = 'resnet34_8x'
                torch.save(student.state_dict(),f"checkpoint/student_{args.model_id}/{args.num_classes}-{name}-{args.epoch_itrs}-{info_loss_coef}.pt")
                torch.save(generator.state_dict(),f"checkpoint/student_{args.model_id}/{args.num_classes}-{name}-{args.epoch_itrs}-{info_loss_coef}-generator.pt")


        # Log Results
        if i % args.log_interval == 0:
            print(f'Train Epoch: {epoch}/{number_epochs} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')
            
            if i == 0:
                with open(args.log_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f,%f\n"%(epoch, loss_G, loss_S))


            # if args.rec_grad_norm and i == 0:

            #     G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
            #     if i == 0:
            #         with open(args.log_dir + "/norm_grad.csv", "a") as f:
            #             f.write("%d,%f,%f,%f\n"%(epoch, G_grad_norm, S_grad_norm, x_true_grad))
                    

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            return best_loss

    return best_loss

def test(args, student = None, generator = None, device = "cuda", test_loader = None, epoch=0):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    myprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    with open(args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n"%(epoch, accuracy))
    acc = correct/len(test_loader.dataset)
    return acc
#     pass

def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return  np.mean(G_grad), np.mean(S_grad)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N', help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)  
    parser.add_argument('--g_iter', type=int, default=1, help = "Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help = "Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lambd', type=float, default=0, metavar='lmbd', help='Generator Info Loss Coefficient(default: 0)')
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=559, help = "Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'],)
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"],)
    parser.add_argument('--steps', nargs='+', default = [0.1, 0.3, 0.5], type=float, help = "Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help = "Fractional decrease in lr")

    parser.add_argument('--dataset', type=int, default=400, choices=[400,600], help='kinetics classes (default: 400)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers, help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=69, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default=None)#'checkpoint/teacher/cifar10-resnet34_8x.pt')
    

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="save_results/cifar10")

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help = 'Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3) 
    

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')
    

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0) 

    parser.add_argument('--store_checkpoints', type=int, default=1)

    parser.add_argument('--student_model', type=str, default='stam',
                        help='Student model architecture (default: resnet18_8x)')

    parser.add_argument('--num_classes', type=int, default=600)
    parser.add_argument('--cuda_num', type=int, default=0)
    args = parser.parse_args()

    args.dataset = args.num_classes
    args.query_budget *=  10**6
    args.query_budget = int(args.query_budget)
    if args.MAZE:

        print("\n"*2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n"*2)
        args.scheduer = "cosine"
        args.loss = "kl"
        args.batch_size = 128
        args.g_iter = 1
        args.d_iter = 5
        args.grad_m = 10
        args.lr_G = 1e-4 
        args.lr_S = 1e-1


    # if args.student_model not in classifiers:
    #     if "wrn" not in args.student_model:
    #         raise ValueError("Unknown model")


    pprint(args, width= 80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)
    torch.autograd.set_detect_anomaly(True)
    
    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    with open(args.log_dir + "/loss.csv", "w") as f:
        f.write("epoch,loss_G,loss_S\n")

    with open(args.log_dir + "/accuracy.csv", "w") as f:
        f.write("epoch,accuracy\n")

    if args.rec_grad_norm:
        with open(args.log_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    with open("latest_experiments.txt", "a") as f:
        f.write(args.log_dir + "\n")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.cuda_num)
    #device = torch.device([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    #print(torch.cuda.device_count())
    #torch.device("cuda:%d"%args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # Preparing checkpoints for the best Student
    global file
    model_dir = f"checkpoint/student_{args.model_id}"; args.model_dir = model_dir
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)  
    file = open(f"{args.model_dir}/logs.txt", "w") 

    print(args)

    args.device = device

    # Eigen values and vectors of the covariance matrix
    # _, test_loader = get_dataloader(args)


    args.normalization_coefs = None
    args.G_activation = torch.tanh

    # num_classes = 10 if args.dataset in ['cifar10', 'svhn'] else 100
    # num_classes = 600;
    # args.num_classes = num_classes

    #if args.model == 'resnet34_8x':
    #    teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    #    if args.dataset == 'svhn':
    #        print("Loading SVHN TEACHER")
    #        args.ckpt = 'checkpoint/teacher/svhn-resnet34_8x.pt'
    #    teacher.load_state_dict( torch.load( args.ckpt, map_location=device) )
    #else:
    #    teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)
    
    

    # teacher.eval()
    # teacher = teacher.to(device)
    # myprint("Teacher restored from %s"%(args.ckpt)) 
    # print(f"\n\t\tTraining with {args.model} as a Target\n") 
    # correct = 0
    # with torch.no_grad():
    #     for i, (data, target) in enumerate(test_loader):
    #         data, target = data.to(device), target.to(device)
    #         output = teacher(data)
    #         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    # accuracy = 100. * correct / len(test_loader.dataset)
    # print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset),accuracy))
    
    assert args.num_classes in [400, 600], "Please enter correct num_classes"
    teacher = None
    if args.num_classes == 600:
        try:
            from movinets import MoViNet
            from movinets.config import _C
        except (ImportError, ModuleNotFoundError):
            raise "MoViNet Import Error!!!"

        teacher = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True).to(device)
        teacher.eval()
        ''' #prev tensorflow
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            from functions import tf_to_torch, torch_to_tf
            hub_url = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3" #/1 gives better on image
            encoder = hub.KerasLayer(hub_url, trainable=False)
            inputs = tf.keras.layers.Input(
                shape=[None, None, None, 3],
                dtype=tf.float32,
                name='image')
            outputs = encoder(dict(image=inputs))
            teacher = tf.keras.Model(inputs, outputs, name='movinet')
        except (ImportError, ModuleNotFoundError):
            pass
        '''
    else:
        try:       
            from swin_transformer_api import SwinT_Kinetics        
            teacher = SwinT_Kinetics()        
            teacher.load_state_dict(torch.load('swint_final_weights.pt'))        
            teacher.to(device)        
            teacher.eval()      
        except ImportError:
            pass

    assert teacher is not None, "Please use the correct environment"

    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes)
    import gc
    gc.collect()
    torch.cuda.empty_cache() 
    # generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32, activation=args.G_activation)
    # generator = network.gan.GeneratorImageOurs(activation=args.G_activation)
    device3 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    generator = network.gan.VideoGenerator(n_channels=3, dim_z_content=79, dim_z_category=400,
                                           dim_z_motion=80, video_length=10, device=device)
    
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    student = student.to(device)
    
    generator = generator.to(device)
    #teacher = teacher.to(device)
    #pdb.set_trace() 
    args.generator = generator
    args.student = student
    args.teacher = teacher

    
    # if args.student_load_path :
    #     # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
    #     student.load_state_dict( torch.load( args.student_load_path ) )
    #     myprint("Student initialized from %s"%(args.student_load_path))
    #     acc = test(args, student=student, generator=generator, device = device, test_loader = test_loader)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m+1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print (f"\nTotal budget: {args.query_budget//1000}k")
    print ("Cost per iterations: ", args.cost_per_iteration)
    print ("Total number of epochs: ", number_epochs)

    student2 = copy.deepcopy(student)
    student2 = student2.to(device)
    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )
    optimizer_S2 = optim.SGD(student2.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9 )

    if args.MAZE:
        optimizer_G = optim.SGD( generator.parameters(), lr=args.lr_G , weight_decay=args.weight_decay, momentum=0.9 )    
    else:
        #optimizer_G = optim.Adam( generator.parameters(), lr=args.lr_G )
        default_lr = 1e-3
        default_mom = 0.9
        optimizer_G = optim.AdamW([
           {'params': generator.modelG.scaleLayers[0][0].parameters(), 'lr':1e-4 },
           {'params': generator.modelG.scaleLayers[0][1].parameters(), 'lr':1e-4 },

           {'params': generator.modelG.scaleLayers[1][0].parameters(), 'lr':5e-4 },
           {'params': generator.modelG.scaleLayers[1][1].parameters(), 'lr':5e-4 },

           {'params': generator.modelG.scaleLayers[2][0].parameters(), 'lr':5e-4 },
           {'params': generator.modelG.scaleLayers[2][1].parameters(), 'lr':5e-4 },

           {'params': generator.modelG.scaleLayers[3][0].parameters(), 'lr':5e-4 },
           {'params': generator.modelG.scaleLayers[3][1].parameters(), 'lr':5e-4 },

           {'params': generator.modelG.scaleLayers[4][0].parameters(), 'lr':1e-3 },
           {'params': generator.modelG.scaleLayers[4][1].parameters(), 'lr':1e-3 },

           {'params': generator.modelG.scaleLayers[5][0].parameters(), 'lr':1e-3 },
           {'params': generator.modelG.scaleLayers[5][1].parameters(), 'lr':1e-3 },

           
        ], lr=default_lr)

        optimizer_RNN = optim.AdamW([
            {'params': generator.recurrent.parameters(), 'lr':1e-3 }
        ])

    lambd = args.lambd
    steps = [4,8,10]#sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_S2 = optim.lr_scheduler.MultiStepLR(optimizer_S2, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
        scheduler_RNN = optim.lr_scheduler.MultiStepLR(optimizer_RNN, steps, args.scale)

    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_S2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_S2, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)
        scheduler_RNN = optim.lr_scheduler.CosineAnnealingLR(optimizer_RNN, number_epochs)


    best_loss = 1
    acc_list = []
    # best_acc = 0
    from tqdm import tqdm
    for epoch in tqdm(range(1, number_epochs + 1)):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_S2.step()
            scheduler_G.step()
            scheduler_RNN.step()

        best_loss = train(args, teacher=teacher, student=student, student2=student2, generator=generator, device=device, optimizer=[optimizer_S,optimizer_S2, optimizer_G, optimizer_RNN], epoch=epoch, best_loss_prev=best_loss, number_epochs = number_epochs, info_loss_coef = lambd)

        # Test
        # acc = test(args, student=student, generator=generator, device = device, test_loader = test_loader, epoch=epoch)
        # acc_list.append(acc)
        # if acc>best_acc:
        #     best_acc = acc
        #     name = 'resnet34_8x'
        #     torch.save(student.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}.pt")
        #     torch.save(generator.state_dict(),f"checkpoint/student_{args.model_id}/{args.dataset}-{name}-generator.pt")
        # # vp.add_scalar('Acc', epoch, acc)
        if args.store_checkpoints:
            torch.save(student.state_dict(), args.log_dir + f"/checkpoint/student-new-{epoch}.pt")
            torch.save(student2.state_dict(), args.log_dir + f"/checkpoint/student2-new-{epoch}.pt")
            torch.save(generator.state_dict(), args.log_dir + f"/checkpoint/generator-new-{epoch}.pt")
    # myprint("Best Acc=%.6f"%best_acc)

    # with open(args.log_dir + "/Max_accuracy = %f"%best_acc, "w") as f:
    #     f.write(" ")

     

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%s.csv'%(args.dataset), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    torch.random.manual_seed(69)
    main()
    """
    from swin_transformer_api import SwinT_Kinetics
    teacher = SwinT_Kinetics()
    student = get_classifier('stam', pretrained=False, num_classes=400)
    generator = network.gan.VideoGenerator(n_channels=3, dim_z_content=79, dim_z_category=400,
                                           dim_z_motion=80, video_length=10, device='cpu')

    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    args = parser.parse_args()
    args.G_activation = torch.tanh
    args.loss = 'kl'
    args.forward_differences = True

    fake, fake_labels, fake_labels_ohe = generator.sample_videos(4, pre_x=1)
    a, b = estimate_gradient_objective(args, teacher, student, fake, fake_labels_ohe, num_classes=400, pre_x=True)
    """
    print()



