import torch
import torch.nn as nn

from Net import *


def build_net(args, shape):
    print("[Build Net]")

    net = EEGNet_net.EEGNet(args, shape)

    # load pretrained parameters
    if args.mode == 'train':
        param = torch.load(f"./pretrained/{args.train_subject[0]-1}/checkpoint/500.tar")
        net.load_state_dict(param['net_state_dict'])

    # test only
    else:
        param = torch.load(f"./tl/{args.train_subject[0]-1}/checkpoint/50.tar")
        net.load_state_dict(param['net_state_dict'])

    # Set GPU
    if args.gpu != 'cpu':
        assert torch.cuda.is_available(), "Check GPU"
        if args.gpu == "multi":
            device = args.gpu
            net = nn.DataParallel(net)
        else:
            device = torch.device(f'cuda:{args.gpu}')
            torch.cuda.set_device(device)
        net.cuda()

    # Set CPU
    else:
        device = torch.device("cpu")

    # Print
    print(f"device: {device}")
    print("")

    return net
