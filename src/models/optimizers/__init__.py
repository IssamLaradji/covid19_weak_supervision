import torch 
from . import sps


def get_optimizer(name, model, exp_dict):
    if name == "adam":
        opt = torch.optim.Adam(
                model.parameters(), lr=exp_dict["lr"], betas=(0.99, 0.999))

    elif name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=exp_dict["lr"])

    elif name == "sps":
        opt = sps.Sps(
            model.parameters(), c=1, momentum=0.6)
    return opt