import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def siamese_loss(f1, f2, sd_label):
    f1 = f1.to(device)
    f2 = f2.to(device)
    sd_label = sd_label.to(device)
    f = F.pairwise_distance(f1, f2, 1).to(device)
    sigmoid_positive = F.sigmoid(f).to(device) + 1e-14
    sigmoid_negative = 1 - sigmoid_positive + 2e-14
    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    loss = torch.mul(sd_label, logistic_positive) + torch.mul(1 - sd_label, logistic_negative)
    return torch.mean(loss)


def contrastive_loss(f1, f2, sd_label):
    f1 = f1.to(device)
    f2 = f2.to(device)
    sd_label = sd_label.to(device)
    f = F.pairwise_distance(f1, f2).to(device)
    dist_positive = f.pow(2)
    dist_negative = torch.max(0 * f, 1 - f).pow(2)
    loss = torch.mul(sd_label, dist_positive) + torch.mul(1 - sd_label, dist_negative)
    return torch.mean(loss)
