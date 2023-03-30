import os
import importlib
import torch.nn as nn
import torch.optim as optim
from utils import read_json


def make_solver(args, net, train_loader, val_loader):
    print("[Make solver]")
    # Set criterion
    criterion = set_criterion(args)

    # Set optimizer
    optimizer, scheduler = set_optimizer(args, net)

    # Set metrics
    log_dict = set_metrics(args)

    # Set solver
    module = importlib.import_module(f"Solver.{args.net}_solver")
    solver = module.Solver(args, net, train_loader, val_loader, criterion, optimizer, scheduler, log_dict)
    print("")
    return solver


def set_criterion(args):
    if args.criterion == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def set_optimizer(args, net):
    if args.opt == "Adam":
        optimizer = optim.Adam(list(net.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    # Set scheduler
    if args.scheduler:
        if args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

        return optimizer, scheduler

    return optimizer, None


def set_metrics(args):
    if args.train_cont_path:
        print("continue log_dict")
        log_dict = read_json(os.path.join(os.path.dirname(os.path.dirname(args.train_cont_path)), "log_dict.json"))
    else:
        log_dict = {f"{phase}_{metric}": [] for phase in ["train", "val"] for metric in args.metrics}
    return log_dict
