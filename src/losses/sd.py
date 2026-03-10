import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SD_loss(f, prior, sd_label):
    prior = prior.to(device)
    f = f.to(device)
    sd_label = sd_label.to(device)
    prior_s = prior ** 2 + (1 - prior) ** 2
    prior_d = 1 - prior_s
    sigmoid_positive = nn.functional.sigmoid(f).to(device) + 1e-14
    sigmoid_negative = 1 - sigmoid_positive + 2e-14
    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    weight = 2 * prior - 1
    loss_sim = prior * logistic_positive - (1 - prior) * logistic_negative
    loss_dis = prior * logistic_negative - (1 - prior) * logistic_positive
    n_sim = torch.sum(sd_label).to(device)
    n_dis = len(sd_label) - n_sim
    loss_similar = prior_s * torch.sum(
        torch.mul(sd_label.view(len(sd_label), -1), loss_sim)
    ) / weight / n_sim
    loss_dissimilar = prior_d * torch.sum(
        torch.mul((1 - sd_label).view(len(sd_label), -1), loss_dis)
    ) / weight / n_dis
    return loss_similar + loss_dissimilar


def logistic(f, label):
    f = f.view(-1, len(label))
    f = torch.mul(label, f)
    sigmoid = F.sigmoid(f) + 1e-14
    logistic_val = -torch.log(sigmoid)
    return torch.mean(logistic_val)
