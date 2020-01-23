import torch.nn as nn


def sum_mse_loss(pred, target):
    """
    :param pred:    Tensor  B,num_stage(3),num_channel(21),46,46
    :param target:
    :return:
    """
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(pred, target)
    return loss / (pred.shape[0] * 46.0 * 46.0)


def ce_loss(pred, target):
    """
    For 0/1 Mask
    :param pred:    Tensor  B,num_stage(3),num_channel(6),46,46
    :param target:  Tensor  B,num_stage(3),num_channel(6),46,46
    :return:
    """
    criterion = nn.BCELoss(reduction='sum')
    return criterion(pred, target) / (pred.shape[0] * 46.0 * 46.0)



