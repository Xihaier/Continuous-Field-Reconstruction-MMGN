import torch

from torch.optim.lr_scheduler import (StepLR, ExponentialLR, ReduceLROnPlateau)


def get_optimizer(params, cfg):
    """
    Set optimizer.
    Args:
        params: model trainable parameters
        cfg: Optimization configuration
    Returns:
        optimizer [torch.optim]
    """
    if cfg.optim_alg == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=cfg.optim_lr)
    elif cfg.optim_alg == "AdamL2":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=cfg.optim_lr, weight_decay=cfg.optim_wd)
    elif cfg.optim_alg == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, params), lr=cfg.optim_lr, weight_decay=cfg.optim_wd)
    return optimizer


def get_scheduler(optimizer, cfg):
    """get learning scheduler.
    Args:
        optimizer [torch.optim]
        cfg: Scheduler configuration.
    Returns:
        scheduler [torch.optim]
    """
    if cfg.name == "StepLR":
        params = cfg.StepLR
        scheduler = StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    elif cfg.name == "ExponentialLR":
        params = cfg.ExponentialLR
        scheduler = ExponentialLR(optimizer, gamma=params.gamma)
    elif cfg.name == "ReduceLROnPlateau":
        params = cfg.ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params.factor, patience=params.patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    return scheduler


def get_loss(loss):
    """
    Set loss.
    Args:
        loss: string.
    Returns:
        Loss function will be use for modeling.
    """
    if loss == "MSELoss":
        criterion = torch.nn.MSELoss(reduction="sum")
    elif loss == "L1Loss":
        criterion = torch.nn.L1Loss(reduction="sum")
    return criterion


def toNumpy(tensor):
    """
    Converts Pytorch tensor to numpy array
    """
    return tensor.detach().cpu().numpy()  