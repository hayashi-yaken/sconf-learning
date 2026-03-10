import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_data import *
from models import *
import scipy 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Sconf_loss(f, prior, sconf, loss_name):
    prior = prior.to(device)
    f=f.to(device)
    sconf=sconf.to(device)
    sigmoid_positive = nn.functional.sigmoid(f).to(device)+1e-14
    sigmoid_negative = 1-sigmoid_positive+2e-14

    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    weight = 2*prior - 1
    coeff_positive = (sconf-(1-prior)).to(device)
    coeff_negative = (prior-sconf).to(device)
    loss_positive = torch.mean(torch.mul(coeff_positive.view(len(sconf),-1), logistic_positive.view(len(sconf), -1))).to(device)/weight
    loss_negative = torch.mean(torch.mul(coeff_negative.view(len(sconf),-1), logistic_negative.view(len(sconf), -1))).to(device)/weight
    if loss_name == 'abs':
        loss_positive = torch.abs(loss_positive).to(device)    #ABS correction
        loss_negative = torch.abs(loss_negative).to(device)    #ABS correction
    elif loss_name == 'nn':
        loss_positive = loss_positive/2+torch.abs(loss_positive/2)   #Non-Negative correction
        loss_negative = loss_negative/2+torch.abs(loss_negative/2)   #Non-Negative correction
    final_loss = loss_positive+loss_negative
    return final_loss


def SD_loss(f, prior, sd_label):
    prior = prior.to(device)
    f = f.to(device)
    sd_label = sd_label.to(device)
    prior_s = prior**2 + (1 - prior)**2
    prior_d = 1 - prior_s
    sigmoid_positive = nn.functional.sigmoid(f).to(device) + 1e-14
    sigmoid_negative = 1 - sigmoid_positive + 2e-14
    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    weight = 2*prior-1
    loss_sim = prior*logistic_positive-(1-prior)*logistic_negative
    loss_dis = prior*logistic_negative-(1-prior)*logistic_positive
    n_sim = torch.sum(sd_label).to(device)
    n_dis = len(sd_label) - n_sim
    loss_similar = prior_s * torch.sum(torch.mul(sd_label.view(len(sd_label),-1),loss_sim))/weight/n_sim
    loss_dissimilar = prior_d * torch.sum(torch.mul((1-sd_label).view(len(sd_label), -1), loss_dis)) / weight / n_dis
    return loss_similar+loss_dissimilar


def logistic(f, label):
    f = f.view(-1,len(label))
    f = torch.mul(label, f)
    sigmoid = F.sigmoid(f)+1e-14
    logistic = -torch.log(sigmoid)
    return torch.mean(logistic)


def siamese_loss(f1, f2, sd_label):
    f1=f1.to(device)
    f2=f2.to(device)
    sd_label=sd_label.to(device)
    f = F.pairwise_distance(f1,f2, 1).to(device)
    sigmoid_positive = F.sigmoid(f).to(device)+1e-14
    sigmoid_negative = 1-sigmoid_positive+2e-14
    logistic_positive = -torch.log(sigmoid_positive).to(device)
    logistic_negative = -torch.log(sigmoid_negative).to(device)
    siamese_loss = torch.mul(sd_label, logistic_positive)+torch.mul(1-sd_label, logistic_negative)
    return torch.mean(siamese_loss)


def contrastive_loss(f1, f2, sd_label):
    f1=f1.to(device)
    f2=f2.to(device)
    sd_label=sd_label.to(device)
    f = F.pairwise_distance(f1, f2).to(device)
    dist_positive = f.pow(2)
    dist_negative = torch.max(0 * f, 1 - f).pow(2)
    contrastive_loss = torch.mul(sd_label, dist_positive)+torch.mul(1-sd_label, dist_negative)
    return torch.mean(contrastive_loss)


def accuracy_check(loader, model):
    sig = F.sigmoid
    num_samples = 0
    total = torch.zeros(1)
    for images, labels in loader:
        labels, images = labels.to(device), images.to(device)
        outputs = model(images).detach()
        sig_outputs = sig(outputs)
        t1 = ((labels ==1 ).view(len(labels),-1)&(sig_outputs>0.5).view(len(labels),-1))|((labels==-1).view(len(labels),-1)&(sig_outputs<0.5).view(len(labels),-1))
        total += torch.sum(t1)
        num_samples += labels.size(0)
    return 100*total / num_samples


def check_sia(test_loader, fp, fn, model):
    num = 0.00
    tot=0.00
    for i, (image,label) in enumerate(test_loader):
        image=image.to(device)
        f=model.sub_forward(image).to(device)
        tot=tot+len(f)
        fp=fp.to(device)
        fn=fn.to(device)
        fp=fp.view(len(fp),-1)
        fn=fn.view(len(fn),-1)
        for i in range(len(f)):
            ff=f[i]
            ff=ff.view(len(ff),-1)
            d1=F.pairwise_distance(ff,fp).to(device)
            d2=F.pairwise_distance(ff,fn).to(device)
            d11=torch.sum(torch.mul(d1,d1))
            d22=torch.sum(torch.mul(d2,d2))
            l=1
            if d11>d22:
                l = -1
            else:
                l = 1
            if l == label[i]:
                num = num+1
    if num > tot-num:
        ans = num
    else:
        ans = tot-num
    ans = ans/tot
    return ans