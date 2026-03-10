import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Sconf_loss(f, prior, sconf, loss_name):
    prior = prior.to(device)
    f = f.to(device)
    sconf = sconf.to(device)
    sigmoid_positive = nn.functional.sigmoid(f).to(device) + 1e-14
    sigmoid_negative = 1 - sigmoid_positive + 2e-14

    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    weight = 2 * prior - 1
    coeff_positive = (sconf - (1 - prior)).to(device)
    coeff_negative = (prior - sconf).to(device)
    loss_positive = torch.mean(
        torch.mul(coeff_positive.view(len(sconf), -1), logistic_positive.view(len(sconf), -1))
    ).to(device) / weight
    loss_negative = torch.mean(
        torch.mul(coeff_negative.view(len(sconf), -1), logistic_negative.view(len(sconf), -1))
    ).to(device) / weight

    if loss_name == 'abs':
        loss_positive = torch.abs(loss_positive).to(device)
        loss_negative = torch.abs(loss_negative).to(device)
    elif loss_name == 'nn':
        loss_positive = loss_positive / 2 + torch.abs(loss_positive / 2)
        loss_negative = loss_negative / 2 + torch.abs(loss_negative / 2)

    return loss_positive + loss_negative
