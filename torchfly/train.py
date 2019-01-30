import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import DataBunch

from typing import Collection, Callable, List


class Learner(object):
    "There can be only one Learner at the same time"
    # class variables
    data: DataBunch
    model: nn.Module
    opt_func: torch.optim
    loss_func: nn.Module
    metrics: Collection
    callbacks: Collection

    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def fit_one_cycle_(self):
        raise NotImplementedError

    def lr_find(self):
        raise NotImplementedError


def train_epoch(model: nn.Module,
                dl: DataLoader,
                opt: torch.optim.Optimizer,
                loss_func: nn.Module):
    model.train()
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_func(model(xb), yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

def validation(model:nn.Module, dl:DataLoader):
    model.eval()
    for xb, yb in dl:
        out = model(xb)
    raise NotImplementedError
    
