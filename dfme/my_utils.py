# from cifar10_models import *
from approximate_gradients import *
# import network
import pytorchvideo
import pytorchvideo.models.resnet as videoresnet

def get_classifier(classifier="x3d_m", pretrained=False, num_classes=400):

    if "x3d" in classifier:
        model = torch.hub.load('facebookresearch/pytorchvideo', classifier, pretrained=False)
        return model
    if "resnet" in classifier: 
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
        if pretrained:
            #TODO: load pretrained on imagenet
            #source: https://github.com/facebookresearch/SlowFast/blob/main/slowfast/utils/checkpoint.py#L142
            pass
        return model
    # if classifier == "wrn-28-10":
    #     net =  wrn(
    #                 num_classes=num_classes,
    #                 depth=28,
    #                 widen_factor=10,
    #                 dropRate=0.3
    #             )
    #     if pretrained:
    #         state_dict = torch.load("cifar100_models/state_dicts/model_best.pt", map_location=device)["state_dict"]
    #         # create new OrderedDict that does not contain `module.`
    #         from collections import OrderedDict
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #         net.load_state_dict(new_state_dict)

    #     return net
    # elif 'wrn' in classifier and 'kt' not in classifier:
    #     depth = int(classifier.split("-")[1])
    #     width = int(classifier.split("-")[2])

    #     net =  wrn(
    #                 num_classes=num_classes,
    #                 depth=depth,
    #                 widen_factor=width
    #             )
    #     if pretrained:
    #         raise ValueError("Cannot be pretrained")
    #     return net
    # elif classifier == "kt-wrn-40-2":
    #     net = WideResNetKT(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
    #     if pretrained:
    #         state_dict = torch.load("cifar10_models/state_dicts/kt_wrn.pt", map_location=device)["state_dict"]
    #         net.load_state_dict(state_dict)
    #     return net
    # elif classifier == "resnet34_8x":
    #     if pretrained:
    #         raise ValueError("Cannot load pretrained resnet34_8x from here")
    #     return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    # elif classifier == "resnet18_8x":
    #     if pretrained:
    #         raise ValueError("Cannot load pretrained resnet18_8x from here")
    #     return network.resnet_8x.ResNet18_8x(num_classes=num_classes)

    # else:
    #     raise NameError('Please enter a valid classifier')


def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad

classifiers = [
    "x3d_m",
    "x3d_s",
    "resnet34",
    "resnet50",
    "resnet101",
    # "resnet34_8x", # Default DFAD
    # "vgg11",
    # "vgg13",
    # "vgg16",
    # "vgg19",
    # "vgg11_bn",
    # "vgg13_bn",
    # "vgg16_bn",
    # "vgg19_bn",
    # "resnet18",
    # "resnet34",
    # "resnet50",
    # "densenet121",
    # "densenet161",
    # "densenet169",
    # "mobilenet_v2",
    # "googlenet",
    # "inception_v3",
    # "wrn-28-10",
    # "resnet18_8x",
    # "kt-wrn-40-2",
]